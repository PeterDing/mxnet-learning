# ResNet
# @paper https://arxiv.org/abs/1512.03385

# mxnet 中有实现 mxnet/gluon/model_zoo/vision/resnet.py

from mxnet.gluon import nn
from mxnet import nd
from mxnet import gluon
from mxnet import init
import utils

# ResNet有效的解决了深度卷积神经网络难训练的问题。
# 这是因为在误差反传的过程中，梯度通常变得越来越小，
# 从而权重的更新量也变小。这个导致远离损失函数的层训练缓慢，
# 随着层数的增加这个现象更加明显。
# ResNet通过增加跨层的连接来解决梯度逐层回传时变小的问题。
#
# {{
# 这个图演示了一个跨层的连接。 http://zh.gluon.ai/_images/residual.svg
# 最底下那层的输入不仅仅是输出给了中间层，而且其与中间层结果相加进入最上层。
# 这样在梯度反传时，最上层梯度可以直接跳过中间层传到最下层，从而避免最下层梯度过小情况。
# }}  -->  为什么?

# ResNet使用跨层通道使得训练非常深的卷积神经网络成为可能。
# 同样它使用很简单的卷积层配置，使得其拓展更加简单。

class Residual(nn.Block):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        strides = 1 if same_shape else 2  ## ??
        self.conv1 = nn.Conv2D(
            channels, kernel_size=3, padding=1, strides=strides)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm()
        if not same_shape:
            self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=strides)

    def forward(self, x):
        out = nd.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return nd.relu(out + x)


# ResNet 18
class ResNet(nn.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        # add name_scope on the outermost Sequential
        with self.name_scope():
            # block 1
            b1 = nn.Conv2D(64, kernel_size=7, strides=2)
            # block 2
            b2 = nn.Sequential()
            b2.add(
                nn.MaxPool2D(pool_size=3, strides=2), Residual(64),
                Residual(64))
            # block 3
            b3 = nn.Sequential()
            b3.add(Residual(128, same_shape=False), Residual(128))
            # block 4
            b4 = nn.Sequential()
            b4.add(Residual(256, same_shape=False), Residual(256))
            # block 5
            b5 = nn.Sequential()
            b5.add(Residual(512, same_shape=False), Residual(512))
            # block 6
            b6 = nn.Sequential()
            b6.add(nn.AvgPool2D(pool_size=3), nn.Dense(num_classes))
            # chain all blocks together
            self.net = nn.Sequential()
            self.net.add(b1, b2, b3, b4, b5, b6)

    def forward(self, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s' % (i + 1, out.shape))
        return out


################################################################
# train

train_data, test_data = utils.load_data_fashion_mnist(batch_size=64, resize=96)

ctx = utils.try_gpu()
net = ResNet(10)
net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05})
utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)
