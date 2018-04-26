# Long short-term memory
# @paper http://www.bioinf.jku.at/publications/older/2604.pdf

# 长短期记忆
## 输入门(input gate)、遗忘门(forget gate)和输出门(output gate)
# I_t = σ(X_t \dot W_xi + H_(t−1) \dot W_hi + b_i)
# F_t = σ(X_t \dot W_xf + H_(t−1) \dot W_hf + b_f)
# O_t = σ(X_t \dot W_xo + H_(t−1) \dot W_ho + b_o)
## 候选细胞
# C̃_t = tanh(X_t \dot W_xc + H_(t−1) \dot W_hc + bc)
## 细胞
# C_t = F_t ⊙ C_(t−1) + I_t ⊙ C̃_t
# 隐含状态
# H_t = O_t ⊙ tanh(C_t)

import zipfile
from mxnet import nd
import utils
ctx = utils.try_gpu()

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


def get_inputs(data):
    return [nd.one_hot(X, vocab_size) for X in data.T]


# params

input_dim = vocab_size
# 隐含状态长度
hidden_dim = 256
output_dim = vocab_size
std = .01


def get_params():
    # 输入门参数
    W_xi = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hi = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_i = nd.zeros(hidden_dim, ctx=ctx)

    # 遗忘门参数
    W_xf = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hf = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_f = nd.zeros(hidden_dim, ctx=ctx)

    # 输出门参数
    W_xo = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_ho = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_o = nd.zeros(hidden_dim, ctx=ctx)

    # 候选细胞参数
    W_xc = nd.random_normal(scale=std, shape=(input_dim, hidden_dim), ctx=ctx)
    W_hc = nd.random_normal(scale=std, shape=(hidden_dim, hidden_dim), ctx=ctx)
    b_c = nd.zeros(hidden_dim, ctx=ctx)

    # 输出层
    W_hy = nd.random_normal(scale=std, shape=(hidden_dim, output_dim), ctx=ctx)
    b_y = nd.zeros(output_dim, ctx=ctx)

    params = [
        W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
        W_hy, b_y
    ]
    for param in params:
        param.attach_grad()
    return params


# model


def lstm_rnn(inputs, state_h, state_c, *params):
    # inputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵
    # H: 尺寸为 batch_size * hidden_dim 矩阵
    # outputs: num_steps 个尺寸为 batch_size * vocab_size 矩阵
    [
        W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
        W_hy, b_y
    ] = params

    H = state_h
    C = state_c
    outputs = []
    for X in inputs:
        I = nd.sigmoid(nd.dot(X, W_xi) + nd.dot(H, W_hi) + b_i)
        F = nd.sigmoid(nd.dot(X, W_xf) + nd.dot(H, W_hf) + b_f)
        O = nd.sigmoid(nd.dot(X, W_xo) + nd.dot(H, W_ho) + b_o)
        C_tilda = nd.tanh(nd.dot(X, W_xc) + nd.dot(H, W_hc) + b_c)
        C = F * C + I * C_tilda
        H = O * nd.tanh(C)
        Y = nd.dot(H, W_hy) + b_y
        outputs.append(Y)
    return (outputs, H, C)


#####################################################################
# train

seq1 = '分开'
seq2 = '不分开'
seq3 = '战争中部队'
seqs = [seq1, seq2, seq3]

utils.train_and_predict_rnn(
    rnn=lstm_rnn,
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
    char_to_idx=char_to_idx,
    is_lstm=True)
