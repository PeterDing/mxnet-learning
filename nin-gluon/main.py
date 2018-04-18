# Network in Network (NiN)
# @paper https://arxiv.org/abs/1312.4400

from mxnet.gluon import nn
from mxnet import gluon
from mxnet import init
import utils


# conv layer 加 dense layer 的形式，重复多次 ( http://zh.gluon.ai/_images/nin.svg )
# 如果把4D矩阵转成2D做全连接，会导致全连接层有过多的参数，
# 所以我用用 1-kernel conv layer 来当作全链接层。
def mlpconv(channels, kernel_size, padding, strides=1, max_pooling=True):
    out = nn.Sequential()
    out.add(
        nn.Conv2D(
            channels=channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation='relu'),
        nn.Conv2D(
            channels=channels,
            kernel_size=1,
            padding=0,
            strides=1,
            activation='relu'),
        nn.Conv2D(
            channels=channels,
            kernel_size=1,
            padding=0,
            strides=1,
            activation='relu'))
    if max_pooling:
        out.add(nn.MaxPool2D(pool_size=3, strides=2))
    return out


##########################################################################
# model build

# 这种“一卷卷到底”最后加一个平均池化层的做法也成为了深度卷积神经网络的常用设计。
net = nn.Sequential()
# add name_scope on the outer most Sequential
with net.name_scope():
    net.add(
        mlpconv(96, 11, 0, strides=4),
        mlpconv(256, 5, 2),
        mlpconv(384, 3, 1),
        nn.Dropout(.5),
        # 目标类为10类 (10个通道)
        mlpconv(10, 3, 1, max_pooling=False),
        # 输入为 batch_size x 10 x 5 x 5, 通过AvgPool2D转成
        # batch_size x 10 x 1 x 1。
        # 我们可以使用 nn.AvgPool2D(pool_size=5),
        # 但更方便是使用全局池化，可以避免估算pool_size大小
        nn.GlobalAvgPool2D(),
        # 转成 batch_size x 10
        nn.Flatten())

##########################################################################
# train

train_data, test_data = utils.load_data_fashion_mnist(
    batch_size=64, resize=224)

ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)
