from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd
import utils

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(256, activation="relu"))
    #  net.add(gluon.nn.Dense(256, activation="sigmoid"))
    #  net.add(gluon.nn.Dense(256, activation="tanh"))
    #  net.add(gluon.nn.Dense(256, activation="softrelu"))
    net.add(gluon.nn.Dense(10))
net.initialize()

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})


def train():
    for epoch in range(10):
        train_loss = 0.
        train_acc = 0.
        for data, label in train_data:
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            trainer.step(batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)

        test_acc = utils.evaluate_accuracy(test_data, net)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" %
              (epoch, train_loss / len(train_data), train_acc / len(train_data),
               test_acc))

def main():
    train()


if __name__ == '__main__':
    main()
