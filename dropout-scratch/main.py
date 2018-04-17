from mxnet import nd
from mxnet import autograd
from mxnet import gluon
import utils

batch_size = 256
train_data, test_data = utils.load_data_fashion_mnist(batch_size)


# dropout
def dropout(X, drop_probability):
    keep_probability = 1 - drop_probability
    assert 0 <= keep_probability <= 1
    # 这种情况下把全部元素都丢弃。
    if keep_probability == 0:
        return X.zeros_like()

    # 随机选择一部分该层的输出作为丢弃元素。
    mask = nd.random.uniform(0, 1.0, X.shape, ctx=X.context) < keep_probability
    # 保证 E[dropout(X)] == X
    scale = 1 / keep_probability
    return mask * X * scale


# model args
num_inputs = 28 * 28
num_outputs = 10

num_hidden1 = 256
num_hidden2 = 256
weight_scale = .01

W1 = nd.random_normal(shape=(num_inputs, num_hidden1), scale=weight_scale)
b1 = nd.zeros(num_hidden1)

W2 = nd.random_normal(shape=(num_hidden1, num_hidden2), scale=weight_scale)
b2 = nd.zeros(num_hidden2)

W3 = nd.random_normal(shape=(num_hidden2, num_outputs), scale=weight_scale)
b3 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2, W3, b3]

for param in params:
    param.attach_grad()

drop_prob1 = 0.
drop_prob2 = 0.


def net(X):
    X = X.reshape((-1, num_inputs))
    # 第一层全连接。
    h1 = nd.relu(nd.dot(X, W1) + b1)
    # 在第一层全连接后添加丢弃层。
    h1 = dropout(h1, drop_prob1)
    # 第二层全连接。
    h2 = nd.relu(nd.dot(h1, W2) + b2)
    # 在第二层全连接后添加丢弃层。
    h2 = dropout(h2, drop_prob2)
    return nd.dot(h2, W3) + b3


softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

learning_rate = .5


def train():
    for epoch in range(5):
        train_loss = 0.
        train_acc = 0.
        for data, label in train_data:
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            utils.SGD(params, learning_rate / batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += utils.accuracy(output, label)

        test_acc = utils.evaluate_accuracy(test_data, net)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" %
              (epoch, train_loss / len(train_data),
               train_acc / len(train_data), test_acc))


def main():
    train()


if __name__ == '__main__':
    main()
