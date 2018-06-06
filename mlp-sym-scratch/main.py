import mxnet as mx
import numpy as np

fname = '../data/letter-recognition.data'
data = np.genfromtxt(fname, delimiter=',')[:, 1:]
label = np.array([ord(l.split(',')[0]) - ord('A') for l in open(fname)])

batch_size = 32
ntrain = int(data.shape[0] * 0.8)
train_iter = mx.io.NDArrayIter(data[:ntrain, :], label[:ntrain], batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(data[ntrain:, :], label[ntrain:], batch_size)

net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
net = mx.sym.Activation(net, name='relu1', act_type="relu")
net = mx.sym.FullyConnected(net, name='fc2', num_hidden=26)
net = mx.sym.SoftmaxOutput(net, name='softmax')

# make model
mod = mx.mod.Module(
    symbol=net, context=mx.cpu(), data_names=['data'], label_names=['softmax_label'])

#
# To train a module, we need to perform following steps:
#
# 1. bind : Prepares environment for the computation by allocating memory.
# 2. init_params : Assigns and initializes parameters.
# 3. init_optimizer : Initializes optimizers. Defaults to sgd.
# 4. metric.create : Creates evaluation metric from input metric name.
# 5. forward : Forward computation.
# 6. update_metric : Evaluates and accumulates evaluation metric on outputs of the last forward computation.
# 7. backward : Backward computation.
# 8. update : Updates parameters according to the installed optimizer
#    and the gradients computed in the previous forward-backward batch.
#

# allocate memory given the input data and label shapes
mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
# initialize parameters by uniform random numbers
mod.init_params(initializer=mx.init.Uniform(scale=.1))
# use SGD with learning rate 0.1 to train
mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1),))
# use accuracy as the metric
metric = mx.metric.create('acc')
# train 5 epochs, i.e. going over the data iter one pass
for epoch in range(5):
    train_iter.reset()
    metric.reset()
    for batch in train_iter:
        mod.forward(batch, is_train=True)  # compute predictions
        mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
        mod.backward()  # compute gradients
        mod.update()  # update parameters
    print('Epoch %d, Training %s' % (epoch, metric.get()))
