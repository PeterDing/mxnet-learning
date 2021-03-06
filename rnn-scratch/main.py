# Recurrent Neural Network

import zipfile
import random
from mxnet import nd
from mxnet import autograd
from mxnet import gluon
from math import exp

# 尝试使用GPU
import sys
sys.path.append('..')
import utils
ctx = utils.try_gpu()
print('Will use', ctx)

# data format

with zipfile.ZipFile('data/jaychou_lyrics.txt.zip', 'r') as zin:
    zin.extractall('data/')

with open('data/jaychou_lyrics.txt') as f:
    corpus_chars = f.read()

corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:20000]

idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
corpus_indices = [char_to_idx[char] for char in corpus_chars]

vocab_size = len(char_to_idx)


def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # 减一是因为label的索引是相应data的索引加一
    num_examples = (len(corpus_indices) - 1) // num_steps
    epoch_size = num_examples // batch_size
    # 随机化样本
    example_indices = list(range(num_examples))
    random.shuffle(example_indices)

    # 返回num_steps个数据
    def _data(pos):
        return corpus_indices[pos:pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size
        batch_indices = example_indices[i:i + batch_size]
        data = nd.array([_data(j * num_steps) for j in batch_indices], ctx=ctx)
        label = nd.array(
            [_data(j * num_steps + 1) for j in batch_indices], ctx=ctx)
        yield data, label


def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):
    corpus_indices = nd.array(corpus_indices, ctx=ctx)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size

    indices = corpus_indices[0:batch_size * batch_len].reshape((batch_size,
                                                                batch_len))
    # 减一是因为label的索引是相应data的索引加一
    epoch_size = (batch_len - 1) // num_steps

    for i in range(epoch_size):
        i = i * num_steps
        data = indices[:, i:i + num_steps]
        label = indices[:, i + 1:i + num_steps + 1]
        yield data, label


def get_inputs(data):
    return [nd.one_hot(X, vocab_size) for X in data.T]


input_dim = vocab_size
# 隐含状态长度
hidden_dim = 256
output_dim = vocab_size
std = .01


def get_params():
    # 隐含层
    W_xh = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hh = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_h = nd.zeros(hidden_dim, ctx=ctx)

    # 输出层
    W_hy = nd.random_normal(scale=std, shape=(hidden_dim, output_dim), ctx=ctx)
    b_y = nd.zeros(output_dim, ctx=ctx)

    params = [W_xh, W_hh, b_h, W_hy, b_y]
    for param in params:
        param.attach_grad()
    return params


# build model
def rnn(inputs, state, *params):
    # inputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵。
    # H: 尺寸为 batch_size * hidden_dim 矩阵。
    # outputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵。
    H = state
    W_xh, W_hh, b_h, W_hy, b_y = params
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return (outputs, H)


# predict
def predict_rnn(rnn,
                prefix,
                num_chars,
                params,
                hidden_dim,
                ctx,
                idx_to_char,
                char_to_idx,
                get_inputs,
                is_lstm=False):
    # 预测以 prefix 开始的接下来的 num_chars 个字符。
    prefix = prefix.lower()
    state_h = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    if is_lstm:
        # 当RNN使用LSTM时才会用到，这里可以忽略。
        state_c = nd.zeros(shape=(1, hidden_dim), ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for i in range(num_chars + len(prefix)):
        X = nd.array([output[-1]], ctx=ctx)
        # 在序列中循环迭代隐含变量。
        if is_lstm:
            # 当RNN使用LSTM时才会用到，这里可以忽略。
            Y, state_h, state_c = rnn(get_inputs(X), state_h, state_c, *params)
        else:
            Y, state_h = rnn(get_inputs(X), state_h, *params)
        if i < len(prefix) - 1:
            next_input = char_to_idx[prefix[i + 1]]
        else:
            next_input = int(Y[0].argmax(axis=1).asscalar())
        output.append(next_input)
    return ''.join([idx_to_char[i] for i in output])


# 梯度剪裁
# 而在循环神经网络的训练中，当每个时序训练数据样本的时序长度num_steps较大或者时刻t较小，
# 目标函数有关t时刻的隐含层变量梯度较容易出现衰减（vanishing）或爆炸（explosion）。
# 梯度剪裁可以解决爆炸的问题，但是对衰减没有效果。
def grad_clipping(params, theta, ctx):
    if theta is not None:
        norm = nd.array([0.0], ctx)
        for p in params:
            norm += nd.sum(p.grad**2)
        norm = nd.sqrt(norm).asscalar()
        if norm > theta:
            for p in params:
                p.grad[:] *= theta / norm


# train
# 困惑度（Perplexity）
# 困惑度可以简单的认为就是对交叉熵做exp运算使得数值更好读。
# 为了解释困惑度的意义，我们先考虑一个完美结果：
# 模型总是把真实的下个字符的概率预测为1。也就是说，对任意的i来说，
# P_target_i=1。这种完美情况下，困惑度值为1。
# 我们再考虑一个基线结果：给定不重复的字符集合W及其字符总数|W|，
# 模型总是预测下个字符为集合W中任一字符的概率都相同。
# 也就是说，对任意的i来说，P_target_i=1/|W|。这种基线情况下，困惑度值为|W|。
# 最后，我们可以考虑一个最坏结果：
# 模型总是把真实的下个字符的概率预测为0。
# 也就是说，对任意的i来说，ptargeti=0。这种最坏情况下，困惑度值为正无穷。
# 任何一个有效模型的困惑度值必须小于预测集中元素的数量。
# 在本例中，困惑度必须小于字典中的字符数|W|。
# 如果一个模型可以取得较低的困惑度的值（更靠近1），通常情况下，该模型预测更加准确。
def train_and_predict_rnn(rnn,
                          is_random_iter,
                          epochs,
                          num_steps,
                          hidden_dim,
                          learning_rate,
                          clipping_theta,
                          batch_size,
                          pred_period,
                          pred_len,
                          seqs,
                          get_params,
                          get_inputs,
                          ctx,
                          corpus_indices,
                          idx_to_char,
                          char_to_idx,
                          is_lstm=False):
    if is_random_iter:
        data_iter = data_iter_random
    else:
        data_iter = data_iter_consecutive
    params = get_params()

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    for e in range(1, epochs + 1):
        # 如使用相邻批量采样，在同一个epoch中，隐含变量只需要在该epoch开始的时候初始化。
        if not is_random_iter:
            state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            if is_lstm:
                # 当RNN使用LSTM时才会用到，这里可以忽略。
                state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
        train_loss, num_examples = 0, 0
        for data, label in data_iter(corpus_indices, batch_size, num_steps,
                                     ctx):
            # 如使用随机批量采样，处理每个随机小批量前都需要初始化隐含变量。
            if is_random_iter:
                state_h = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
                if is_lstm:
                    # 当RNN使用LSTM时才会用到，这里可以忽略。
                    state_c = nd.zeros(shape=(batch_size, hidden_dim), ctx=ctx)
            with autograd.record():
                # outputs 尺寸：(batch_size, vocab_size)
                if is_lstm:
                    # 当RNN使用LSTM时才会用到，这里可以忽略。
                    outputs, state_h, state_c = rnn(
                        get_inputs(data), state_h, state_c, *params)
                else:
                    outputs, state_h = rnn(get_inputs(data), state_h, *params)
                # 设t_ib_j为i时间批量中的j元素:
                # label 尺寸：（batch_size * num_steps）
                # label = [t_0b_0, t_0b_1, ..., t_1b_0, t_1b_1, ..., ]
                label = label.T.reshape((-1, ))
                # 拼接outputs，尺寸：(batch_size * num_steps, vocab_size)。
                outputs = nd.concat(*outputs, dim=0)
                # 经上述操作，outputs和label已对齐。
                loss = softmax_cross_entropy(outputs, label)
            loss.backward()

            grad_clipping(params, clipping_theta, ctx)
            utils.SGD(params, learning_rate)

            train_loss += nd.sum(loss).asscalar()
            num_examples += loss.size

        if e % pred_period == 0:
            print("Epoch %d. Perplexity %f" % (e,
                                               exp(train_loss / num_examples)))
            for seq in seqs:
                print(' - ',
                      predict_rnn(rnn, seq, pred_len, params, hidden_dim, ctx,
                                  idx_to_char, char_to_idx, get_inputs,
                                  is_lstm))
            print()


######################################################################
epochs = 200
num_steps = 35
learning_rate = 0.1
batch_size = 32

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

seq1 = '分开'
seq2 = '不分开'
seq3 = '战争中部队'
seqs = [seq1, seq2, seq3]

train_and_predict_rnn(
    rnn=rnn,
    is_random_iter=True,
    epochs=200,
    num_steps=35,
    hidden_dim=hidden_dim,
    learning_rate=0.2,
    clipping_theta=5,
    batch_size=32,
    pred_period=20,
    pred_len=100,
    seqs=seqs,
    get_params=get_params,
    get_inputs=get_inputs,
    ctx=ctx,
    corpus_indices=corpus_indices,
    idx_to_char=idx_to_char,
    char_to_idx=char_to_idx)
