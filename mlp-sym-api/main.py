import logging
logging.getLogger().setLevel(logging.INFO)
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

# reset train_iter to the beginning
train_iter.reset()

# fit the module
mod.fit(
    train_iter,
    eval_data=val_iter,
    optimizer='sgd',
    optimizer_params={'learning_rate': 0.1},
    eval_metric='acc',
    num_epoch=8)

y = mod.predict(val_iter)
assert y.shape == (4000, 26)

score = mod.score(val_iter, ['acc'])
print("Accuracy score is %f" % (score[0][1]))

# Save and Load
# construct a callback function to save checkpoints
model_prefix = 'mx_mlp'
checkpoint = mx.callback.do_checkpoint(model_prefix)
# def checkpoint_struct(iter_no, sym, arg, aux):
#     """The checkpoint callback function."""
#     pass

mod = mx.mod.Module(symbol=net)
mod.fit(train_iter, num_epoch=5, epoch_end_callback=checkpoint)

# Load 1
sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, 3)
assert sym.tojson() == net.tojson()

# assign the loaded parameters to the module
mod.set_params(arg_params, aux_params)

# Load 2
mod = mx.mod.Module(symbol=sym)
mod.fit(train_iter, num_epoch=21, arg_params=arg_params, aux_params=aux_params, begin_epoch=3)
