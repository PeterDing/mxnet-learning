# Fully Convolutional Networks (FCN)
# @paper https://arxiv.org/abs/1411.4038

# 语义分割
# 我们已经学习了如何识别图片里面的主要物体，和找出里面物体的边框。
# 语义分割则在之上更进一步，它对每个像素预测它是否只是背景，还是属于哪个我们感兴趣的物体。
# 图片分割经常只需要利用像素之间的相似度即可，而语义分割则需要详细的类别标号。
# 这也是为什么称其为语义的原因。

# 本章我们将介绍利用卷积神经网络解决语义分割的一个开创性工作之一：全链接卷积网络。

# 数据集
# VOC2012是一个常用的语义分割数据集。
# > http://host.robots.ox.ac.uk/pascal/VOC/voc2012/

import os
import tarfile
from mxnet import gluon
from mxnet import image
from mxnet import nd
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
import numpy as np

import utils

data_root = '../data'
voc_root = data_root + '/VOCdevkit/VOC2012'
url = ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012' '/VOCtrainval_11-May-2012.tar')
sha1 = '4e443f8a2eca6b1dac8a6c57641b67dd40621a49'

if os.path.exists(voc_root):
    os.makedirs(voc_root)
    fname = gluon.utils.download(url, data_root, sha1_hash=sha1)

    if not os.path.isfile(voc_root + '/ImageSets/Segmentation/train.txt'):
        with tarfile.open(fname, 'r') as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, data_root)


def read_images(root=voc_root, train=True):
    txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    n = len(images)
    data, label = [None] * n, [None] * n
    for i, fname in enumerate(images):
        data[i] = image.imread('%s/JPEGImages/%s.jpg' % (root, fname))
        label[i] = image.imread('%s/SegmentationClass/%s.png' % (root, fname))
    return data, label


# 图片的宽度基本是500，但高度各不一样。
# 为了能将多张图片合并成一个批量来加速计算，我们需要输入图片都是同样的大小。
# 之前我们通过imresize来将他们调整成同样的大小。
# 但在语义分割里，我们需要对标注做同样的变化来达到像素级别的匹配。
# 但调整大小将改变像素颜色，使得再将它们映射到物体类别变得困难。
# 这里我们仅仅使用剪切来解决这个问题。
# 就是说对于输入图片，我们随机剪切出一个固定大小的区域，然后对标号图片做同样位置的剪切。
def rand_crop(data, label, height, width):
    data, rect = image.random_crop(data, (width, height))
    label = image.fixed_crop(label, *rect)
    return data, label


# 接下来我们列出每个物体和背景对应的RGB值
classes = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'potted plant',
    'sheep', 'sofa', 'train', 'tv/monitor'
]
# RGB color for each class
colormap = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128],
            [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0],
            [192, 128, 0], [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
            [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

# 这样给定一个标号图片，我们就可以将每个像素对应的物体标号找出来。
cm2lbl = np.zeros(256**3)
for i, cm in enumerate(colormap):
    # !! 注意，这里 cm2lbl 元素的值域是 [0, len(classes) - 1]
    cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def image2label(im):
    data = im.astype('int32').asnumpy()
    # data[:, :, 0].shape = data.shape[:-1]
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]

    # a.shape = (i1, i2, i3, ..., in)
    # b.shape = (j1, j2, ..., jm)
    # a[b].shape = [b.shape, a.shape[1:]]
    # a[b][b.shape:] = a[b[b.shape]]
    #
    # 如此，cm2lbl[idx] 中每个像素对应者他的类别序号。
    return nd.array(cm2lbl[idx])


# 现在我们可以定义数据读取了。
# 每一次我们将图片和标注随机剪切到要求的形状，并将标注里每个像素转成对应的标号。
# 简单起见我们将小于要求大小的图片全部过滤掉了。
rgb_mean = nd.array([0.485, 0.456, 0.406])  # 怎么得到的？？
rgb_std = nd.array([0.229, 0.224, 0.225])  # 怎么得到的？？


def normalize_image(data):
    return (data.astype('float32') / 255 - rgb_mean) / rgb_std


class VOCSegDataset(gluon.data.Dataset):

    def _filter(self, images):
        return [
            im for im in images
            if (im.shape[0] >= self.crop_size[0] and im.shape[1] >= self.crop_size[1])
        ]

    def __init__(self, train, crop_size):
        self.crop_size = crop_size
        data, label = read_images(train=train)
        data = self._filter(data)
        self.data = [normalize_image(im) for im in data]
        self.label = self._filter(label)
        print('Read ' + str(len(self.data)) + ' examples')

    def __getitem__(self, idx):
        data, label = rand_crop(self.data[idx], self.label[idx], *self.crop_size)
        data = data.transpose((2, 0, 1))
        label = image2label(label)
        return data, label

    def __len__(self):
        return len(self.data)


# 我们采用320×480的大小用来训练，注意到这个比前面我们使用的224×224要大上很多。
# 但是同样我们将长宽都定义成了32的整数倍。
# height x width
input_shape = (320, 480)
voc_train = VOCSegDataset(True, input_shape)
voc_test = VOCSegDataset(False, input_shape)

# 最后定义批量读取。可以看到跟之前的不同是批量标号不再是一个向量，而是一个三维数组。
batch_size = 64
train_data = gluon.data.DataLoader(voc_train, batch_size, shuffle=True, last_batch='discard')
test_data = gluon.data.DataLoader(voc_test, batch_size, last_batch='discard')

# 全连接卷积网络
# 在数据的处理过程我们看到语义分割跟前面介绍的应用的主要区别在于，
# 预测的标号不再是一个或者几个数字，而是每个像素都需要有标号。
# 在卷积神经网络里，我们通过卷积层和池化层逐渐减少数据长宽但同时增加通道数。
# 例如ResNet18里，我们先将输入长宽减少32倍，由3×224×224的图片转成512×7×7的输出，
# 应该全局池化层变成512长向量，然后最后通过全链接层转成一个长度为n的输出向量，
# 这里n是类数，既num_classes。但在这里，对于输出为3×320×480的图片，我们需要输出是n×320×480，
# 就是每个输入像素都需要预测一个长度为n的向量。
#
# 全连接卷积网络（FCN）的提出是基于这样一个观察。
# 假设f是一个卷积层，而且y=f(x)。那么在反传求导时，∂f(y)会返回一个跟x一样形状的输出。
# 卷积是一个对偶函数，就是∂^2 f=f。那么如果我们想得到跟输入一样的输入，那么定义g=∂f，这样g(f(x))就能达到我们想要的。
# 具体来说，我们定义一个卷积转置层（transposed convolutional, 也经常被错误的叫做deconvolutions），
# 它就是将卷积层的forward和backward函数兑换。
#
# 另外一点要注意的是，在最后的卷积层我们同样使用平化层（nn.Flattern）或者（全局）池化层来使得方便使用之后的全连接层作为输出。
# 但是这样会损害空间信息，而这个对语义分割很重要。
# 一个解决办法是去掉不需要的池化层，并将全连接层替换成1×1卷基层。
#
# 所以给定一个卷积网络，FCN主要做下面的改动:
#
# 替换全连接层成1×1卷基
# 去掉过于损失空间信息的池化层，例如全局池化
# 最后接上卷积转置层来得到需要大小的输出
# 为了训练更快，通常权重会初始化称预先训练好的权重
#
pretrained_net = models.resnet18_v2(pretrained=True)
# 我们看到feature模块最后两层是GlobalAvgPool2D和Flatten，都是我们不需要的。
# 所以我们定义一个新的网络，它复制除了最后两层的features模块的权重。
net = nn.HybridSequential()
for layer in pretrained_net.features[:-2]:
    net.add(layer)
# 然后接上一个通道数等于类数的1×1卷积层。
# 注意到net已经将输入长宽减少了32倍。那么我们需要接入一个strides=32的卷积转置层。
# 我们使用一个比stides大两倍的kernel，然后补上适当的填充。
num_classes = len(classes)
with net.name_scope():
    net.add(
        nn.Conv2D(num_classes, kernel_size=1),
        nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16, strides=32))


# 训练
# 训练的时候我们需要初始化新添加的两层。
# 我们可以随机初始化，但实际中发现将卷积转置层初始化成双线性差值函数可以使得训练更容易。
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return nd.array(weight)


# 下面代码演示这样的初始化等价于对图片进行双线性差值放大。
# from matplotlib import pyplot as plt
#
# x = train_images[0]
# print('Input', x.shape)
# x = x.astype('float32').transpose((2,0,1)).expand_dims(axis=0)/255
#
# conv_trans = nn.Conv2DTranspose(
#     3, in_channels=3, kernel_size=8, padding=2, strides=4)
# conv_trans.initialize()
# conv_trans(x)
# conv_trans.weight.set_data(bilinear_kernel(3, 3, 8))
#
#
# y = conv_trans(x)
# y = y[0].clip(0,1).transpose((1,2,0))
# print('Output', y.shape)
#
# plt.imshow(y.asnumpy())
# plt.show()

conv_trans = net[-1]
conv_trans.initialize(init=init.Zero())
net[-2].initialize(init=init.Xavier())

x = nd.zeros((batch_size, 3, input_shape[0], input_shape[1]))
net(x)

shape = conv_trans.weight.data().shape
conv_trans.weight.set_data(bilinear_kernel(*shape[0:3]))

# 这时候我们可以真正开始训练了。
# 值得一提的是我们使用卷积转置层的通道来预测像素的类别。
# 所以在做softmax和预测的时候我们需要使用通道这个维度，既维度1。
# 所以在SoftmaxCrossEntropyLoss里加入了额外了axis=1选项。
# 其他的部分跟之前的训练一致。
loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)

ctx = utils.try_all_gpus()
net.collect_params().reset_ctx(ctx)

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1, 'wd': 1e-3})

utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=10)


# 预测
# 预测函数跟之前的图片分类预测类似，但跟上面一样，主要不同在于我们需要在axis=1上做argmax。
# 同时我们定义image2label的反函数，它将预测值转成图片。
def predict(im):
    data = normalize_image(im)
    data = data.transpose((2, 0, 1)).expand_dims(axis=0)
    yhat = net(data.as_in_context(ctx[0]))
    pred = nd.argmax(yhat, axis=1)
    return pred.reshape((pred.shape[1], pred.shape[2]))


def label2image(pred):
    # x.shape 是图片的，元素是类别序号。
    x = pred.astype('int32').asnumpy()
    cm = nd.array(colormap).astype('uint8')
    # cm[x].shape = [x.shape, 3]     heigh, weight, column
    return nd.array(cm[x, :])


# 我们读取前几张测试图片并对其进行预测。
test_images, test_labels = read_images(train=False)

n = 6
imgs = []
for i in range(n):
    x = test_images[i]
    pred = label2image(predict(x))
    imgs += [x, pred, test_labels[i]]

# utils.show_images(imgs, nrows=n, ncols=3, figsize=(6,10))
