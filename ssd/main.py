import os
import time
import numpy as np
from mxnet import gluon
from mxnet import image
from mxnet import nd
from mxnet import metric
from mxnet import init
from mxnet import gpu, cpu
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.contrib.ndarray import MultiBoxPrior
from mxnet.contrib.ndarray import MultiBoxTarget
from mxnet.contrib.ndarray import MultiBoxDetection

# data download
root_url = (
    'https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/pikachu/')
data_dir = '../data/pikachu/'
dataset = {
    'train.rec': 'e6bcb6ffba1ac04ff8a9b1115e650af56ee969c8',
    'train.idx': 'dcf7318b2602c06428b9988470c731621716c393',
    'val.rec': 'd6c33f799b4d058e82f2cb5bd9a976f69d72d520'
}

if not os.listdir(data_dir):
    for k, v in dataset.items():
        gluon.utils.download(root_url + k, data_dir + k, sha1_hash=v)

########################################################################
# data reader
data_shape = 256
batch_size = 32


def get_iterators(data_shape, batch_size):
    class_names = ['pikachu']
    num_class = len(class_names)
    train_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec=data_dir + 'train.rec',
        path_imgidx=data_dir + 'train.idx',
        shuffle=True,
        mean=True,
        rand_crop=1,
        min_object_covered=0.95,
        max_attempts=200)
    val_iter = image.ImageDetIter(
        batch_size=batch_size,
        data_shape=(3, data_shape, data_shape),
        path_imgrec=data_dir + 'val.rec',
        shuffle=False,
        mean=True)
    return train_iter, val_iter, class_names, num_class


train_data, test_data, class_names, num_class = get_iterators(
    data_shape, batch_size)

###########
# SSD模型 #
###########

# 锚框：默认的边界框
# shape: batch x channel x height x weight
## n = 40
## x = nd.random.uniform(shape=(1, 3, n, n))
##
## y = MultiBoxPrior(x, sizes=[.5,.25,.1], ratios=[1,2,.5])
##
## boxes = y.reshape((n, n, -1, 4))
## print(boxes.shape)
## # The first anchor box centered on (20, 20)
## # its format is (x_min, y_min, x_max, y_max)
## boxes[20, 20, 0, :]


# 预测物体类别
# 对每一个锚框我们需要预测它是不是包含了我们感兴趣的物体，
# 还是只是背景。这里我们使用一个3×3的卷积层来做预测，
# 加上pad=1使用它的输出和输入一样。
# 同时输出的通道数是num_anchors*(num_classes+1)，
# 每个通道对应一个锚框对某个类的置信度。
# 假设输出是Y，那么对应输入中第n个样本的第(i,j)像素的置信值是在Y[n,:,i,j]里。
# 具体来说，对于以(i,j)为中心的第a个锚框，
#
# 通道 a*(num_class+1) 是其只包含背景的分数
# 通道 a*(num_class+1)+1+b 是其包含第b个物体的分数
#
# 我们定义个一个这样的类别分类器函数：
def class_predictor(num_anchors, num_classes):
    """return a layer to predict classes"""
    return nn.Conv2D(num_anchors * (num_classes + 1), 3, padding=1)


# 预测边界框
# 因为真实的边界框可以是任意形状，我们需要预测如何从一个锚框变换成真正的边界框。
# 这个变换可以由一个长为4的向量来描述。
# 同上一样，我们用一个有num_anchors * 4通道的卷积。
# 假设输出是Y，那么对应输入中第 n 个样本的第 (i,j) 像素为中心的锚框的转换在Y[n,:,i,j]里。
# 具体来说，对于第a个锚框，它的变换在a*4到a*4+3通道里。
def box_predictor(num_anchors):
    """return a layer to predict delta locations"""
    return nn.Conv2D(num_anchors * 4, 3, padding=1)


# 减半模块
# 我们定义一个卷积块，它将输入特征的长宽减半，以此来获取多尺度的预测。
# 它由两个Conv-BatchNorm-Relu组成，我们使用填充为1的3×3卷积使得输入和输入有同样的长宽，
# 然后再通过跨度为2的最大池化层将长宽减半。
def down_sample(num_filters):
    """stack two Conv-BatchNorm-Relu blocks and then a pooling layer
    to halve the feature size"""
    out = nn.HybridSequential()
    for _ in range(2):
        out.add(nn.Conv2D(num_filters, 3, strides=1, padding=1))
        out.add(nn.BatchNorm(in_channels=num_filters))
        out.add(nn.Activation('relu'))
    out.add(nn.MaxPool2D(2))
    return out


# 合并来自不同层的预测输出
# 前面我们提到过SSD的一个重要性质是它会在多个层同时做预测。
# 每个层由于长宽和锚框选择不一样，导致输出的数据形状会不一样。
# 为了之后处理简单，我们将不同层的输入合并成一个输出。
# 首先我们将通道移到最后的维度，然后将其展成2D数组。
# 因为第一个维度是样本个数，所以不同输出之间是不变，
# 我们可以将所有输出在第二个维度上拼接起来。
def flatten_prediction(pred):
    return pred.transpose(axes=(0, 2, 3, 1)).flatten()


def concat_predictions(preds):
    return nd.concat(*preds, dim=1)


# 主体网络
# 主体网络用来从原始像素抽取特征。
# 通常前面介绍的用来图片分类的卷积神经网络，例如ResNet，都可以用来作为主体网络。
# 这里为了示范，我们简单叠加几个减半模块作为主体网络。
def body():
    out = nn.HybridSequential()
    for nfilters in [16, 32, 64]:
        out.add(down_sample(nfilters))
    return out


# 创建一个玩具SSD模型
# 现在我们可以创建一个玩具SSD模型了。我们称之为玩具是因为这个网络不管是层数还是锚框个数都比较小，仅仅适合之后我们之后使用的一个小数据集。但这个模型不会影响我们介绍SSD。
# 这个网络包含四块。主体网络，三个减半模块，以及五个物体类别和边框预测模块。
# 其中预测分别应用在在主体网络输出，减半模块输出，和最后的全局池化层上。
def toy_ssd_model(num_anchors, num_classes):
    downsamplers = nn.Sequential()
    for _ in range(3):
        downsamplers.add(down_sample(128))

    class_predictors = nn.Sequential()
    box_predictors = nn.Sequential()
    for _ in range(5):
        class_predictors.add(class_predictor(num_anchors, num_classes))
        box_predictors.add(box_predictor(num_anchors))

    model = nn.Sequential()
    model.add(body(), downsamplers, class_predictors, box_predictors)
    return model


# 计算预测
# 给定模型和每层预测输出使用的锚框大小和形状，我们可以定义前向函数。
def toy_ssd_forward(x, model, sizes, ratios, verbose=False):
    body, downsamplers, class_predictors, box_predictors = model
    anchors, class_preds, box_preds = [], [], []

    # feature extraction
    x = body(x)

    for i in range(5):

        # predict
        anchors.append(MultiBoxPrior(x, sizes=sizes[i], ratios=ratios[i]))
        class_preds.append(flatten_prediction(class_predictors[i](x)))
        box_preds.append(flatten_prediction(box_predictors[i](x)))
        if verbose:
            print('Predict scale', i, x.shape, 'with', anchors[-1].shape[1],
                  'anchors')

        # down sample
        if i < 3:
            x = downsamplers[i](x)
        elif i == 3:
            x = nd.Pooling(
                x,
                global_pool=True,
                pool_type='max',
                kernel=(x.shape[2], x.shape[3]))

    # concat data
    return (concat_predictions(anchors), concat_predictions(class_preds),
            concat_predictions(box_preds))


# 完整的模型
class ToySSD(gluon.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ToySSD, self).__init__(**kwargs)
        # anchor box sizes and ratios for 5 feature scales
        self.sizes = [[.2, .272], [.37, .447], [.54, .619], [.71, .79],
                      [.88, .961]]
        self.ratios = [[1, 2, .5]] * 5
        self.num_classes = num_classes
        self.verbose = verbose
        num_anchors = len(self.sizes[0]) + len(self.ratios[0]) - 1
        # use name_scope to guard the names
        with self.name_scope():
            self.model = toy_ssd_model(num_anchors, num_classes)

    def forward(self, x):
        anchors, class_preds, box_preds = toy_ssd_forward(
            x, self.model, self.sizes, self.ratios, verbose=self.verbose)
        # it is better to have class predictions reshaped for softmax computation
        class_preds = class_preds.reshape(shape=(0, -1, self.num_classes + 1))
        return anchors, class_preds, box_preds


# 训练
# 之前的教程我们主要是关注分类。对于分类的预测结果和真实的标号，我们通过交叉熵来计算他们的差异。
# 但物体检测里我们需要预测边框。
# 这里我们先引入一个概率来描述两个边框的距离。

# IoU：交集除并集
# 我们知道判断两个集合的相似度最常用的衡量叫做Jaccard距离，
# 给定集合 A 和 B，它的定义是 J(A,B) = |A∩B| / |A∪B|
# 大的值表示两个边框很相似，而小的值则表示不相似。


# 损失函数
# 虽然每张图片里面通常只有几个标注的边框，但SSD会生成大量的锚框。
# 可以想象很多锚框都不会框住感兴趣的物体，就是说跟任何对应感兴趣物体的表框的IoU都小于某个阈值。
# 这样就会产生大量的负类锚框，或者说对应标号为0的锚框。对于这类锚框有两点要考虑的：
#
# 1. 边框预测的损失函数不应该包括负类锚框，因为它们并没有对应的真实边框
# 2. 因为负类锚框数目可能远多于其他，我们可以只保留其中的一些。
#    而且是保留那些目前预测最不确信它是负类的，就是对类0预测值排序，
#    选取数值最小的哪一些困难的负类锚框。
# 我们可以使用MultiBoxTarget来完成上面这两个操作。
# 它返回三个NDArray，分别是
# 1. 预测的边框跟真实边框的偏移，大小是batch_size x (num_anchors*4)
# 2. 用来遮掩不需要的负类锚框的掩码，大小跟上面一致
# 3. 锚框的真实的标号，大小是batch_size x num_anchors
def training_targets(anchors, class_preds, labels):
    class_preds = class_preds.transpose(axes=(0, 2, 1))
    return MultiBoxTarget(anchors, labels, class_preds)


# 对于分类问题，最常用的损失函数是之前一直使用的交叉熵。
# 这里我们定义一个类似于交叉熵的损失，不同于交叉熵的定义 log(pj)，
# 这里 j 是真实的类别，且 pj 是对于的预测概率。
# 我们使用一个被称之为关注损失的函数，给定正的γ和α，它的定义是
# −α(1 − p_j)^γ * log(p_j)
# 这个自定义的损失函数可以简单通过继承gluon.loss.Loss来实现。
class FocalLoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

    def hybrid_forward(self, F, output, label):
        output = F.softmax(output)
        pj = output.pick(label, axis=self._axis, keepdims=True)
        loss = -self._alpha * ((1 - pj)**self._gamma) * pj.log()
        return loss.mean(axis=self._batch_axis, exclude=True)


cls_loss = FocalLoss()


# 对于边框的预测是一个回归问题。
# 通常可以选择平方损失函数（L2损失）f(x)=x2。
# 但这个损失对于比较大的误差的惩罚很高。
# 我们可以采用稍微缓和一点绝对损失函数（L1损失）f(x)=|x|，
# 它是随着误差线性增长，而不是平方增长。
# 但这个函数在0点处导数不唯一，因此可能会影响收敛。
# 一个通常的解决办法是在0点附近使用平方函数使得它更加平滑。
# 它被称之为平滑L1损失函数。它通过一个参数σ来控制平滑的区域：
#
# f(x) = (σx)^2 / 2         if x < 1/σ^2
# f(x) = |x| − 0.5 / σ^2    otherwise
# 我们同样通过继承Loss来定义这个损失。
# 同时它接受一个额外参数mask，这是用来屏蔽掉不需要被惩罚的负例样本。
class SmoothL1Loss(gluon.loss.Loss):
    def __init__(self, batch_axis=0, **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)

    def hybrid_forward(self, F, output, label, mask):
        loss = F.smooth_l1((output - label) * mask, scalar=1.0)
        return loss.mean(self._batch_axis, exclude=True)


box_loss = SmoothL1Loss()

# 评估测量
# 对于分类好坏我们可以沿用之前的分类精度。
# 评估边框预测的好坏的一个常用是是平均绝对误差。
# 记得在线性回归我们使用了平均平方误差。
# 但跟上面对损失函数的讨论一样，平方误差对于大的误差给予过大的值，从而数值上过于敏感。
# 平均绝对误差就是将二次项替换成绝对值，具体来说就是预测的边框和真实边框在4个维度上的差值的绝对值。

cls_metric = metric.Accuracy(
)  # \text{accuracy}(y, \\hat{y}) = \\frac{1}{n} \\sum_{i=0}^{n-1} \text{1}(\\hat{y_i} == y_i)
box_metric = metric.MAE()  # \frac{\sum_i^n |y_i - \hat{y}_i|}{n}

# 初始化模型和训练器
#  ctx = gpu(0)
ctx = cpu(0)
# the CUDA implementation requres each image has at least 3 lables.
# Padd two -1 labels for each instance
train_data.reshape(label_shape=(3, 5))
train_data = test_data.sync_label_shape(train_data)

#  num_class = 2
net = ToySSD(num_class)
net.initialize(init.Xavier(magnitude=2), ctx=ctx)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {
    'learning_rate': 0.1,
    'wd': 5e-4
})

# 训练模型
# 训练函数跟前面的不一样在于网络会有多个输出，而且有两个损失函数。
for epoch in range(30):
    # reset data iterators and metrics
    train_data.reset()
    cls_metric.reset()
    box_metric.reset()
    tic = time.time()
    for i, batch in enumerate(train_data):
        x = batch.data[0].as_in_context(ctx)
        y = batch.label[0].as_in_context(ctx)
        with autograd.record():
            anchors, class_preds, box_preds = net(x)
            box_target, box_mask, cls_target = training_targets(
                anchors, class_preds, y)
            # losses
            loss1 = cls_loss(class_preds, cls_target)
            loss2 = box_loss(box_preds, box_target, box_mask)
            loss = loss1 + loss2
        loss.backward()
        trainer.step(batch_size)
        # update metrics
        cls_metric.update([cls_target], [class_preds.transpose((0, 2, 1))])
        box_metric.update([box_target], [box_preds * box_mask])

    _a1, _a2 = cls_metric.get()
    _b1, _b2 = box_metric.get()
    print('Epoch %2d, train %s %.2f, %s %.5f, time %.1f sec' %
          (epoch, _a1, _a2, _b1, _b2, time.time() - tic))

# 预测
# 在预测阶段，我们希望能把图片里面所有感兴趣的物体找出来。
# 我们先定一个数据读取和预处理函数。
rgb_mean = nd.array([123, 117, 104])


def process_image(fname):
    with open(fname, 'rb') as f:
        im = image.imdecode(f.read())
    # resize to data_shape
    data = image.imresize(im, data_shape, data_shape)
    # minus rgb mean
    data = data.astype('float32') - rgb_mean
    # convert to batch x channel x height xwidth
    return data.transpose((2, 0, 1)).expand_dims(axis=0), im


# 然后我们跟训练那样预测表框和其对应的物体。
# 但注意到因为我们对每个像素都会生成数个锚框，这样我们可能会预测出大量相似的表框，
# 从而导致结果非常嘈杂。
# 一个办法是对于IoU比较高的两个表框，我们只保留预测执行度比较高的那个。
# 这个算法（称之为non maximum suppression）在MultiBoxDetection里实现了。
# 预测函数会输出所有边框，每个边框由[class_id, confidence, xmin, ymin, xmax, ymax]表示。
# 其中class_id=-1表示要么这个边框被预测只含有背景，或者被去重掉了。
# 下面我们实现预测函数：
def predict(x):
    anchors, cls_preds, box_preds = net(x.as_in_context(ctx))
    # cls_preds: (batch, anchor_size, num_class)
    cls_probs = nd.SoftmaxActivation(
        cls_preds.transpose((0, 2, 1)), mode='channel')

    return MultiBoxDetection(
        cls_probs, box_preds, anchors, force_suppress=True, clip=False)


# 最后我们将预测出置信度超过某个阈值的边框画出来：
threshold = 0.5
x, im = process_image('../img/pikachu.jpg')
out = predict(x)
for row in out:
    row = row.asnumpy()
    class_id, score = int(row[0]), row[1]
    if class_id < 0 or score < threshold:
        continue
    box = row[2:6] * np.array([im.shape[0], im.shape[1]] * 2)
    # display box
