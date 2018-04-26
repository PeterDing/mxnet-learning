import math
import os
import time
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn
import utils
context = utils.try_gpu()

import zipfile
with zipfile.ZipFile('data/ptb.zip', 'r') as zin:
    zin.extractall('data/')


# 建立词语索引
class Dictionary(object):
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = []

    def add_word(self, word):
        if word not in self.word_to_idx:
            self.idx_to_word.append(word)
            self.word_to_idx[word] = len(self.idx_to_word) - 1
        return self.word_to_idx[word]

    def __len__(self):
        return len(self.idx_to_word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(path + 'train.txt')
        self.valid = self.tokenize(path + 'valid.txt')
        self.test = self.tokenize(path + 'test.txt')

    def tokenize(self, path):
        assert os.path.exists(path)
        # 将词语添加至词典。
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)
        # 将文本转换成词语索引的序列（NDArray格式）。
        with open(path, 'r') as f:
            indices = np.zeros((tokens, ), dtype='int32')
            idx = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    indices[idx] = self.dictionary.word_to_idx[word]
                    idx += 1
        return mx.nd.array(indices, dtype='int32')


data = 'data/ptb/ptb.'
corpus = Corpus(data)
vocab_size = len(corpus.dictionary)


# 循环神经网络模型库
class RNNModel(gluon.Block):
    """循环神经网络模型库"""

    def __init__(self,
                 mode,
                 vocab_size,
                 embed_dim,
                 hidden_dim,
                 num_layers,
                 dropout=0.5,
                 **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(
                vocab_size, embed_dim, weight_initializer=mx.init.Uniform(0.1))
            if mode == 'rnn_relu':
                self.rnn = rnn.RNN(
                    hidden_dim,
                    num_layers,
                    activation='relu',
                    dropout=dropout,
                    input_size=embed_dim)
            elif mode == 'rnn_tanh':
                self.rnn = rnn.RNN(
                    hidden_dim,
                    num_layers,
                    activation='tanh',
                    dropout=dropout,
                    input_size=embed_dim)
            elif mode == 'lstm':
                self.rnn = rnn.LSTM(
                    hidden_dim,
                    num_layers,
                    dropout=dropout,
                    input_size=embed_dim)
            elif mode == 'gru':
                self.rnn = rnn.GRU(
                    hidden_dim,
                    num_layers,
                    dropout=dropout,
                    input_size=embed_dim)
            else:
                raise ValueError("Invalid mode %s. Options are rnn_relu, "
                                 "rnn_tanh, lstm, and gru" % mode)

            self.decoder = nn.Dense(vocab_size, in_units=hidden_dim)
            self.hidden_dim = hidden_dim

    def forward(self, inputs, state):
        emb = self.drop(self.encoder(inputs))
        output, state = self.rnn(emb, state)
        output = self.drop(output)
        decoded = self.decoder(output.reshape((-1, self.hidden_dim)))
        return decoded, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


model_name = 'rnn_relu'

embed_dim = 100
hidden_dim = 100
num_layers = 2
lr = 1.0
clipping_norm = 0.2
epochs = 1
batch_size = 32
num_steps = 5
dropout_rate = 0.2
eval_period = 500


# 批量采样
# 相邻批量采样
def batchify(data, batch_size):
    """数据形状 (num_batches, batch_size)"""
    num_batches = data.shape[0] // batch_size
    data = data[:num_batches * batch_size]
    data = data.reshape((batch_size, num_batches)).T
    return data


train_data = batchify(corpus.train, batch_size).as_in_context(context)
val_data = batchify(corpus.valid, batch_size).as_in_context(context)
test_data = batchify(corpus.test, batch_size).as_in_context(context)

model = RNNModel(model_name, vocab_size, embed_dim, hidden_dim, num_layers,
                 dropout_rate)
model.collect_params().initialize(mx.init.Xavier(), ctx=context)
trainer = gluon.Trainer(model.collect_params(), 'sgd', {
    'learning_rate': lr,
    'momentum': 0,
    'wd': 0
})
loss = gluon.loss.SoftmaxCrossEntropyLoss()


def get_batch(source, i):
    seq_len = min(num_steps, source.shape[0] - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len]
    return data, target.reshape((-1, ))


# 从计算图分离隐含状态
def detach(state):
    if isinstance(state, (tuple, list)):
        state = [i.detach() for i in state]
    else:
        state = state.detach()
    return state


# 训练和评价模型
def model_eval(data_source):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(
        func=mx.nd.zeros, batch_size=batch_size, ctx=context)
    for i in range(0, data_source.shape[0] - 1, num_steps):
        data, target = get_batch(data_source, i)
        output, hidden = model(data, hidden)
        L = loss(output, target)
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return total_L / ntotal


############################################################
# train
def train():
    for epoch in range(epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(
            func=mx.nd.zeros, batch_size=batch_size, ctx=context)
        for ibatch, i in enumerate(
                range(0, train_data.shape[0] - 1, num_steps)):
            data, target = get_batch(train_data, i)
            # 从计算图分离隐含状态。
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # 梯度裁剪。需要注意的是，这里的梯度是整个批量的梯度。
            # 因此我们将clipping_norm乘以num_steps和batch_size。
            gluon.utils.clip_global_norm(
                grads, clipping_norm * num_steps * batch_size)

            trainer.step(batch_size)
            total_L += mx.nd.sum(L).asscalar()

            if ibatch % eval_period == 0 and ibatch > 0:
                cur_L = total_L / num_steps / batch_size / eval_period
                print('[Epoch %d Batch %d] loss %.2f, perplexity %.2f' %
                      (epoch + 1, ibatch, cur_L, math.exp(cur_L)))
                total_L = 0.0

        val_L = model_eval(val_data)

        print('[Epoch %d] time cost %.2fs, validation loss %.2f, validation '
              'perplexity %.2f' % (epoch + 1, time.time() - start_time, val_L,
                                   math.exp(val_L)))


train()
test_L = model_eval(test_data)
print('Test loss %.2f, test perplexity %.2f' % (test_L, math.exp(test_L)))
