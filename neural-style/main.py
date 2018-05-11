# 样式迁移
# @paper https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

# Gatys等人开创性的通过匹配卷积神经网络的中间层输出来训练出合成图片。它的流程如下所示：
# https://zh.gluon.ai/_images/neural-style2.svg
#
# 1. 我们首先挑选一个卷积神经网络来提取特征。
#    我们选择它的特定层来匹配样式，特定层来匹配内容。
#    示意图中我们选择层1,2,4作为样式层，层3作为内容层。
# 2. 输入样式图片并保存样式层输出，记第 i 层输出为 s_i
# 3. 输入内容图片并保存内容层输出，记第 i 层输出为 c_i
# 4. 初始化合成图片 x 为随机值或者其他更好的初始值。
#    然后进行迭代使得用 x 抽取的特征能够匹配上 s_i 和 c_i。
#    具体来说，我们如下迭代直到收敛。
# 5. 输入 x 计算样式层和内容层输出，记第 i 层输出为 y_i
# 6. 使用样式损失函数来计算 y_i 和 s_i 的差异
# 7. 使用内容损失函数来计算 y_i 和 c_i 的差异
# 8. 对损失求和并对输入 x 求导，记导数为 g
# 9. 更新 x， 例如 x=x−ηg

# 内容损失函数使用通常回归用的均方误差。
# 对于样式，我们可以将它看成是像素点在每个通道的统计分布。
# 例如要匹配两张图片的颜色，我们的一个做法是匹配这两张图片在RGB这三个通道上的直方图。
# 更一般的，假设卷积层的输出格式是c×h×w，既channels x height x width。
# 那么我们可以把它变形成 c×hw 的2D数组，并将它看成是一个维度为c 的随机变量采样到的 hw 个点。
# 所谓的样式匹配就是使得两个 c 维随机变量统计分布一致。
#
# 匹配统计分布常用的做法是冲量匹配，就是说使得他们有一样的均值，协方差，和其他高维的冲量。
# 为了计算简单起见，我们这里假设卷积输出已经是均值为0了，而且我们只匹配协方差。
# 也就是说，样式损失函数就是对 s_i 和 y_i 计算 Gram 矩阵然后应用均方误差
#
# styleloss(s_i,y_i)= \frac{1}{c^2 hw} ‖s_i s^T_i − y_i y^T_i‖_F
#
# 这里假设我们已经将 s_i 和 y_i 变形成了 c×hw 的2D矩阵了。

from time import time
from mxnet import autograd
from mxnet import image
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
import utils
import matplotlib.pyplot as plt

# 数据
style_img = image.imread('../img/style1.jpg')
content_img = image.imread('../img/cont1.jpg')

# style_img.shape = (heigh, width)
image_shape = (style_img.shape[1] // 3, style_img.shape[0] // 3)
y0 = content_img.shape[0] // 2 - style_img.shape[0] // 2
x0 = content_img.shape[1] // 2 - style_img.shape[1] // 2
content_img = image.fixed_crop(content_img, x0, y0, style_img.shape[1], style_img.shape[0])

# 跟前面教程一样我们定义预处理和后处理函数，它们将原始图片进行归一化并转换成卷积网络接受的输入格式，
# 和还原成能展示的图片格式。
rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])


def preprocess(img, image_shape):
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32') / 255 - rgb_mean) / rgb_std
    return img.transpose((2, 0, 1)).expand_dims(axis=0)


def postprocess(img):
    img = img[0].as_in_context(rgb_std.context)
    return (img.transpose((1, 2, 0)) * rgb_std + rgb_mean).clip(0, 1)


# 模型
# 我们使用原论文使用的VGG 19模型。并下载在Imagenet上训练好的权重。
pretrained_net = models.vgg19(pretrained=True)

# 回忆VGG这一章里，我们使用五个卷积块vgg_block来构建网络。快之间使用nn.MaxPool2D来做间隔。
# 我们有很多种选择来使用某些层作为样式和内容的匹配层。
# 通常越靠近输入层越容易匹配内容和样式的细节信息，越靠近输出则越倾向于语义的内容和全局的样式。
# 这里我们按照原论文使用每个卷积块的第一个卷基层输出来匹配样式，和第四个块中的最后一个卷积层来匹配内容。
# 根据pretrained_net的输出我们记录下这些层对应的位置。
style_layers = [0, 5, 10, 19, 28]
content_layers = [25]


# 因为只需要使用中间层的输出，我们构建一个新的网络，它只保留我们需要的层。
def get_net(pretrained_net, content_layers, style_layers):
    net = nn.Sequential()
    for i in range(max(content_layers + style_layers) + 1):
        net.add(pretrained_net.features[i])
    return net


net = get_net(pretrained_net, content_layers, style_layers)


# 给定输入x，简单使用net(x)只能拿到最后的输出，而这里我们还需要net的中间层输出。
# 因此我们我们逐层计算，并保留需要的输出。
def extract_features(x, content_layers, style_layers):
    contents = []
    styles = []
    for i in range(len(net)):
        x = net[i](x)
        if i in style_layers:
            styles.append(x)
        if i in content_layers:
            contents.append(x)
    return contents, styles


# 损失函数
#
#  内容匹配
# 内容匹配是一个典型的回归问题，我们将来使用均方误差来比较内容层的输出。
def content_loss(yhat, y):
    return (yhat - y).square().mean()


#  样式匹配
# 样式匹配则是通过拟合Gram矩阵。我们先定义它的计算：
def gram(x):
    c = x.shape[1]
    n = x.size / x.shape[1]
    y = x.reshape((c, int(n)))
    return nd.dot(y, y.T) / n


#  和对应的损失函数。
# 对于要匹配的样式图片它的样式输出在训练中不会改变，我们将提前计算好它的Gram矩阵来作为输入使得计算加速。
def style_loss(yhat, gram_y):
    return (gram(yhat) - gram_y).square().mean()


#  降噪
# 当使用靠近输出层的高层输出来拟合时，经常可以观察到学到的图片里面有大量高频噪音。
# 这个有点类似老式天线电视机经常遇到的白噪音。
# 有多种方法来降噪，例如可以加入模糊滤镜，或者使用总变差降噪（Total Variation Denoising）。
#
# 假设 xi,j 表示像素 (i,j)，那么我们加入下面的损失函数，它使得邻近的像素值相似：
#
# \sum_{i,j} (|x_{i,j} − x_{i+1,j}| + |x_{i,j} − x_{i,j+1}|)
def tv_loss(yhat):
    return 0.5 * ((yhat[:, :, 1:, :] - yhat[:, :, :-1, :]).abs().mean() +
                  (yhat[:, :, :, 1:] - yhat[:, :, :, :-1]).abs().mean())


# 权重
# 总损失函数是上述三个损失函数的加权和。
# 通过调整权重值我们可以控制学到的图片是否保留更多样式，更多内容，还是更加干净。
# 注意到样式匹配中我们使用了5个层的输出，我们对靠近输入的层给予比较大的权重。
channels = [net[l].weight.shape[0] for l in style_layers]
style_weights = [1e4 / n**2 for n in channels]
content_weights = [1]
tv_weight = 10


# 我们可以使用nd.add_n来将多个损失函数的输出按权重加起来。
def sum_loss(loss, preds, truths, weights):
    return nd.add_n(*[w * loss(yhat, y) for w, yhat, y in zip(weights, preds, truths)])


# 训练
# 首先我们定义两个函数，他们分别对源内容图片和源样式图片提取特征。
def get_contents(image_shape):
    content_x = preprocess(content_img, image_shape).copyto(ctx)
    content_y, _ = extract_features(content_x, content_layers, style_layers)
    return content_x, content_y


def get_styles(image_shape):
    style_x = preprocess(style_img, image_shape).copyto(ctx)
    _, style_y = extract_features(style_x, content_layers, style_layers)
    style_y = [gram(y) for y in style_y]
    return style_x, style_y


# 训练过程跟之前的主要的主要不同在于
#
# 1. 这里我们的损失函数更加复杂。
# 2. 我们只对输入进行更新，这个意味着我们需要对输入x预先分配了梯度。
# 3. 我们可能会替换匹配内容和样式的层，和调整他们之间的权重，来得到不同风格的输出。
#    这里我们对梯度做了一般化，使得不同参数下的学习率不需要太大变化。
# 4. 仍然使用简单的梯度下降，但每n次迭代我们会减小一次学习率
def train(x, max_epochs, lr, lr_decay_epoch=200):
    tic = time()
    for i in range(max_epochs):
        with autograd.record():
            content_py, style_py = extract_features(x, content_layers, style_layers)
            content_L = sum_loss(content_loss, content_py, content_y, content_weights)
            style_L = sum_loss(style_loss, style_py, style_y, style_weights)
            tv_L = tv_weight * tv_loss(x)
            loss = style_L + content_L + tv_L

        loss.backward()
        x.grad[:] /= x.grad.abs().mean() + 1e-8
        x[:] -= lr * x.grad
        # add sync to avoid large mem usage
        nd.waitall()

        print('epoch', i)

        if i and i % 20 == 0:
            print('batch %3d, content %.2f, style %.2f, '
                  'TV %.2f, time %.1f sec' % (i, content_L.asscalar(), style_L.asscalar(),
                                              tv_L.asscalar(), time() - tic))
            tic = time()

        if i and i % lr_decay_epoch == 0:
            lr *= 0.1
            print('change lr to ', lr)

    return x


ctx = utils.try_gpu()
net.collect_params().reset_ctx(ctx)

content_x, content_y = get_contents(image_shape)
style_x, style_y = get_styles(image_shape)

x = content_x.copyto(ctx)
x.attach_grad()

y = train(x, 500, 0.1)
plt.imsave('result.png', postprocess(y).asnumpy())
