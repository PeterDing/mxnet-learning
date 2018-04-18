from mxnet.gluon import nn
from mxnet import gluon
from mxnet import init
import utils

# @paper https://arxiv.org/pdf/1409.1556.pdf

# 多个 conv layers 加一个 Pooling
def vgg_block(num_convs, channels):
    out = nn.Sequential()
    for _ in range(num_convs):
        out.add(
            nn.Conv2D(
                channels=channels, kernel_size=3, padding=1,
                activation='relu'))
    out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out


# 顺序添加多个 vgg_block
def vgg_stack(architecture):
    out = nn.Sequential()
    for (num_convs, channels) in architecture:
        out.add(vgg_block(num_convs, channels))
    return out


###############################################################
# model and params
num_outputs = 10
architecture = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
net = nn.Sequential()
# add name_scope on the outermost Sequential
# 8 conv layer + 3 denses = VGG 11
# 13 conv layer + 3 denses = VGG 16
# 16 conv layer + 3 denses = VGG 19
with net.name_scope():
    net.add(
        vgg_stack(architecture), nn.Flatten(), nn.Dense(
            4096, activation="relu"), nn.Dropout(.5),
        nn.Dense(4096, activation="relu"), nn.Dropout(.5),
        nn.Dense(num_outputs))

###############################################################
# train
train_data, test_data = utils.load_data_fashion_mnist(batch_size=64, resize=96)

ctx = utils.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())

loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.05})
utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=1)
