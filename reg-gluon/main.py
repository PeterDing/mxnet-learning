from mxnet import ndarray as nd
from mxnet import autograd
from mxnet import gluon
import mxnet as mx

#  import matplotlib as mpl
#  mpl.rcParams['figure.dpi'] = 120
#  import matplotlib.pyplot as plt

num_train = 20
num_test = 100
num_inputs = 200

true_w = nd.ones((num_inputs, 1)) * 0.01
true_b = 0.05

X = nd.random.normal(shape=(num_train + num_test, num_inputs))
y = nd.dot(X, true_w) + true_b
y += .01 * nd.random.normal(shape=y.shape)

X_train, X_test = X[:num_train, :], X[num_train:, :]
y_train, y_test = y[:num_train], y[num_train:]

batch_size = 1
dataset_train = gluon.data.ArrayDataset(X_train, y_train)
data_iter_train = gluon.data.DataLoader(
    dataset_train, batch_size, shuffle=True)

square_loss = gluon.loss.L2Loss()


def test(net, X, y):
    return square_loss(net(X), y).mean().asscalar()
    #return np.mean(square_loss(net(X, *params), y).asnumpy())


def train(weight_decay):
    epochs = 10
    learning_rate = 0.005
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Dense(1))
    net.collect_params().initialize(mx.init.Normal(sigma=1))

    # 注意到这里 'wd'
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {
        'learning_rate': learning_rate,
        'wd': weight_decay
    })

    train_loss = []
    test_loss = []
    for e in range(epochs):
        for data, label in data_iter_train:
            with autograd.record():
                output = net(data)
                loss = square_loss(output, label)
            loss.backward()
            trainer.step(batch_size)

        train_loss.append(test(net, X_train, y_train))
        test_loss.append(test(net, X_test, y_test))
    #  plt.plot(train_loss)
    #  plt.plot(test_loss)
    #  plt.legend(['train', 'test'])
    #  plt.show()
    print('learned w[:10]:', net[0].weight.data()[:, :10], 'learned b:',
          net[0].bias.data())
    print('train_loss:', train_loss)
    print('test_loss:', test_loss)


def main():
    train(0)
    print()
    train(5)


if __name__ == '__main__':
    main()
