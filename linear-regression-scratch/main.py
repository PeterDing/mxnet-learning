import random
from mxnet import ndarray as nd
from mxnet import autograd

from utils import SGD

num_inputs = 2
num_examples = 1000

true_w = [2, -3.4]
true_b = 4.2

X = nd.random_normal(shape=(num_examples, num_inputs))
y = true_w[0] * X[:, 0] + true_w[1] * X[:, 1] + true_b
y += .01 * nd.random_normal(shape=y.shape)

batch_size = 10


def data_iter():
    # 产生一个随机索引
    idx = list(range(num_examples))
    random.shuffle(idx)
    for i in range(0, num_examples, batch_size):
        j = nd.array(idx[i:min(i + batch_size, num_examples)])
        yield nd.take(X, j), nd.take(y, j)


w = nd.random_normal(shape=(num_inputs, 1))
b = nd.zeros((1, ))
params = [w, b]

for param in params:
    param.attach_grad()


def net(X):
    return nd.dot(X, w) + b


def square_loss(yhat, y):
    # 注意这里我们把y变形成yhat的形状来避免矩阵形状的自动转换
    return (yhat - y.reshape(yhat.shape))**2


def train():
    epochs = 5
    learning_rate = .001
    niter = 0
    losses = []
    moving_loss = 0
    smoothing_constant = .01

    # 训练
    for e in range(epochs):
        total_loss = 0

        for data, label in data_iter():
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            SGD(params, learning_rate)
            total_loss += nd.sum(loss).asscalar()

            # 记录每读取一个数据点后，损失的移动平均值的变化；
            niter += 1
            curr_loss = nd.mean(loss).asscalar()
            moving_loss = (1 - smoothing_constant) * moving_loss + (
                smoothing_constant) * curr_loss

            # correct the bias from the moving averages
            est_loss = moving_loss / (1 - (1 - smoothing_constant)**niter)

            if (niter + 1) % 100 == 0:
                losses.append(est_loss)
                print(
                    "Epoch %s, batch %s. Moving avg of loss: %s. Average loss: %f"
                    % (e, niter, est_loss, total_loss / num_examples))
                #  plot(losses, X)


def main():
    train()
    print('over')
    print(w, b)


if __name__ == '__main__':
    main()
