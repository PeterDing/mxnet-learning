from mxnet.gluon import nn
from mxnet import autograd
from mxnet import gluon
from mxnet import nd
import utils

net = nn.Sequential()
with net.name_scope():
    # 第一层卷积
    net.add(nn.Conv2D(channels=20, kernel_size=5))
    ### 添加了批量归一化层
    net.add(nn.BatchNorm(axis=1))  # axis 指 channel 再的 axis
    net.add(nn.Activation(activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    # 第二层卷积
    net.add(nn.Conv2D(channels=50, kernel_size=3))
    ### 添加了批量归一化层
    net.add(nn.BatchNorm(axis=1))
    net.add(nn.Activation(activation='relu'))
    net.add(nn.MaxPool2D(pool_size=2, strides=2))
    net.add(nn.Flatten())
    # 第一层全连接
    net.add(nn.Dense(128, activation="relu"))
    # 第二层全连接
    net.add(nn.Dense(10))

ctx = utils.try_gpu()
net.initialize(ctx=ctx)
batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2})


def train():
    for epoch in range(5):
        train_loss = 0.
        train_acc = 0.
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data.as_in_context(ctx))
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)
        test_acc = utils.evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" %
              (epoch, train_loss / len(train_data),
               train_acc / len(train_data), test_acc))


def main():
    train()


if __name__ == '__main__':
    main()
