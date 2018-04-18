# DenseNet
# @paper https://arxiv.org/abs/1608.06993

# DenseNet 结构和 ResNet 一样，只是来自跳层的输出不是通过加法（+）而是拼接（concat）来跟目前层的输出合并。

from mxnet import nd
from mxnet.gluon import nn
from mxnet import gluon
from mxnet import init
import utils


def conv_block(channels):
    out = nn.Sequential()
    out.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(channels, kernel_size=3, padding=1))
    return out


# Dense Block
# DenseNet的卷积块使用ResNet改进版本的BN->Relu->Conv。
# 每个卷积的输出通道数被称之为growth_rate，这是因为假设输出为in_channels，
# 而且有layers层，那么输出的通道数就是in_channels+growth_rate*layers。
class DenseBlock(nn.Block):
    def __init__(self, layers, growth_rate, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for i in range(layers):
            self.net.add(conv_block(growth_rate))

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = nd.concat(x, out, dim=1)
        return x


# Transition Block
# 因为使用拼接的缘故，每经过一次拼接输出通道数可能会激增。
# 为了控制模型复杂度，这里引入一个过渡块，它不仅把输入的长宽减半，
# 同时也使用1×1卷积来改变通道数。
def transition_block(channels):
    out = nn.Sequential()
    out.add(nn.BatchNorm(), nn.Activation('relu'),
            nn.Conv2D(channels, kernel_size=1),
            nn.AvgPool2D(pool_size=2, strides=2))
    return out


# body
# 121层的DenseNet
init_channels = 64
growth_rate = 32
block_layers = [6, 12, 24, 16]
num_classes = 10


def dense_net():
    net = nn.Sequential()
    # add name_scope on the outermost Sequential
    with net.name_scope():
        # first block
        net.add(
            nn.Conv2D(init_channels, kernel_size=7, strides=2, padding=3),
            nn.BatchNorm(), nn.Activation('relu'),
            nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        # dense blocks
        channels = init_channels
        for i, layers in enumerate(block_layers):
            net.add(DenseBlock(layers, growth_rate))
            channels += layers * growth_rate
            if i != len(block_layers) - 1:
                net.add(transition_block(channels // 2))
        # last block
        net.add(
            nn.BatchNorm(),
            nn.Activation('relu'),
            nn.AvgPool2D(pool_size=1),
            nn.Flatten(),
            nn.Dense(num_classes))
    return net


#################################################################
# train

train_data, test_data = utils.load_data_fashion_mnist(batch_size=64, resize=32)

ctx = utils.try_gpu()
net = dense_net()
net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)
