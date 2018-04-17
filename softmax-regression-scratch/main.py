import sys
from mxnet import gluon
from mxnet import ndarray as nd
from mxnet import autograd

from utils import SGD, accuracy, evaluate_accuracy


def transform(data, label):
    return data.astype('float32') / 255, label.astype('float32')


mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)

batch_size = 256
train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)

# init model args
num_inputs = 784
num_outputs = 10

# W.shape = (784, 10)
W = nd.random_normal(shape=(num_inputs, num_outputs))
# b.shape = (10,)
b = nd.random_normal(shape=num_outputs)

params = [W, b]
for param in params:
    param.attach_grad()

# define model
def softmax(X):
    # X.shape = (256, 10)
    exp = nd.exp(X)
    # 假设exp是矩阵，这里对行进行求和，并要求保留axis 1，
    # 就是返回 (nrows, 1) 形状的矩阵
    # partition.shape = (256, 1)
    partition = exp.sum(axis=1, keepdims=True)
    # a[i,j] = exp[i,j] / partition[i,1]
    a = exp / partition
    return a 


def net(X):
    # nd.dot(X.reshape((-1, num_inputs)).shape = (256, 784)
    # W.shape = (784, 10)
    # so, nd.dot(X.reshape((-1, num_inputs)), W).shape = (256, 10)
    # (256, 10) + (10,) add (10,) with every row of (256, 10)
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)


# loss function
def cross_entropy(yhat, y):
    # nd.pick(input, index)[i,j] = 
    return -nd.pick(nd.log(yhat), y)


def train():
    learning_rate = .1

    for epoch in range(10):
        train_loss = 0.
        train_acc = 0.
        for data, label in train_data:
            with autograd.record():
                output = net(data)
                loss = cross_entropy(output, label)
            loss.backward()
            # 将梯度做平均，这样学习率会对batch size不那么敏感
            SGD(params, learning_rate / batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += accuracy(output, label)

        test_acc = evaluate_accuracy(test_data, net)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" %
              (epoch, train_loss / len(train_data),
               train_acc / len(train_data), test_acc))


def main():
    train()
    print(W, b)


if __name__ == '__main__':
    main()
