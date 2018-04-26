# Gated Recurrent Neural Networks
# @paper https://arxiv.org/abs/1406.1078

# 门控循环神经网络（gated recurrent neural networks）的提出，
# 是为了更好地捕捉时序数据中间隔较大的依赖关系。
# 其中，门控循环单元（gated recurrent unit，简称GRU）是一种常用的门控循环神经网络。

# 门控循环单元
## 重置门(reset gate)和更新门(update gate)
# R_t = σ(X_t \dot W_xr + H_(t−1) \dot W_hr + b_r)   # 上一时刻隐含状态H_(t−1)
# Z_t = σ(X_t \dot W_xz + H_(t−1) \dot W_hz + b_z)
# σ(x)=1/(1+exp(−x)), sigmoid function
## 候选隐含状态
# H̃_t = tanh(X_t \dot W_xh + R_t ⊙ H_(t−1) \dot W_hh + b_h)
## 隐含状态
# H_t = Z_t ⊙ H_(t−1) + (1 − Z_t) ⊙ H̃_t
# 更新门可以控制过去的隐含状态在当前时刻的重要性。
# 如果更新门一直近似1，过去的隐含状态将一直通过时间保存并传递至当前时刻。
# 这个设计可以应对循环神经网络中的梯度衰减问题，并更好地捕捉时序数据中间隔较大的依赖关系。

# 重置门有助于捕捉时序数据中短期的依赖关系。
# 更新门有助于捕捉时序数据中长期的依赖关系。

import zipfile
from mxnet import nd
import utils

ctx = utils.try_gpu()

with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip', 'r') as zin:
    zin.extractall('../data/')

with open('../data/jaychou_lyrics.txt') as f:
    corpus_chars = f.read()

corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:20000]

idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
corpus_indices = [char_to_idx[char] for char in corpus_chars]

vocab_size = len(char_to_idx)


def get_inputs(data):
    return [nd.one_hot(X, vocab_size) for X in data.T]


# params

input_dim = vocab_size
# 隐含状态长度
hidden_dim = 256
output_dim = vocab_size
std = .01


def get_params():
    # 隐含层
    W_xz = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hz = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_z = nd.zeros(hidden_dim, ctx=ctx)

    W_xr = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hr = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_r = nd.zeros(hidden_dim, ctx=ctx)

    W_xh = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hh = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_h = nd.zeros(hidden_dim, ctx=ctx)

    # 输出层
    W_hy = nd.random_normal(scale=std, shape=(hidden_dim, output_dim), ctx=ctx)
    b_y = nd.zeros(output_dim, ctx=ctx)

    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hy, b_y]
    for param in params:
        param.attach_grad()
    return params


def gru_rnn(inputs, H, *params):
    # inputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵
    # H: 尺寸为 batch_size * hidden_dim 矩阵
    # outputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hy, b_y = params
    outputs = []
    for X in inputs:
        Z = nd.sigmoid(nd.dot(X, W_xz) + nd.dot(H, W_hz) + b_z)
        R = nd.sigmoid(nd.dot(X, W_xr) + nd.dot(H, W_hr) + b_r)
        H_tilda = nd.tanh(nd.dot(X, W_xh) + R * nd.dot(H, W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return (outputs, H)


##################################################################
# train

seq1 = '分开'
seq2 = '不分开'
seq3 = '战争中部队'
seqs = [seq1, seq2, seq3]

utils.train_and_predict_rnn(
    rnn=gru_rnn,
    is_random_iter=False,
    epochs=200,
    num_steps=35,
    hidden_dim=hidden_dim,
    learning_rate=0.2,
    clipping_norm=5,
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
