# Neural Machine Translation with Attention
# @paper https://www.aclweb.org/anthology/D14-1179
# @paper https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf
# @paper https://arxiv.org/abs/1409.0473

import sys
import mxnet as mx
from mxnet import autograd, gluon, nd
from mxnet.gluon import nn, rnn, Block
from mxnet.contrib import text

from io import open
import collections
import datetime

# 下面定义一些特殊字符。
# PAD (padding)符号使每个序列等长
# BOS (beginning of sequence)符号表示序列的开始
# EOS (end of sequence)符号表示序列的结束。
PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'

# 模型参数
# 我们在编码器和解码器中分别使用了一层和两层的循环神经网络。
epochs = 50
epoch_period = 10

learning_rate = 0.005
# 每步计算梯度时使用的样本个数
batch_size = 2
# 输入序列的最大长度（含句末添加的EOS字符）。
max_seq_len = 5
# 输出序列的最大长度（含句末添加的EOS字符）。
max_output_len = 20

encoder_num_layers = 1
decoder_num_layers = 2

encoder_drop_prob = 0.1
decoder_drop_prob = 0.1

encoder_hidden_dim = 256
decoder_hidden_dim = 256
alignment_dim = 25

ctx = mx.cpu(0)


# 读取数据
def read_data(max_seq_len):
    input_tokens = []
    output_tokens = []
    input_seqs = []
    output_seqs = []

    with open('../data/fr-en-small.txt') as f:
        lines = f.readlines()
        for line in lines:

            input_seq, output_seq = line.rstrip().split('\t')
            cur_input_tokens = input_seq.split(' ')
            cur_output_tokens = output_seq.split(' ')

            if len(cur_input_tokens) < max_seq_len and \
                            len(cur_output_tokens) < max_seq_len:
                input_tokens.extend(cur_input_tokens)
                # 句末附上EOS符号。
                cur_input_tokens.append(EOS)
                # 添加PAD符号使每个序列等长（长度为max_seq_len）。
                while len(cur_input_tokens) < max_seq_len:
                    cur_input_tokens.append(PAD)
                input_seqs.append(cur_input_tokens)

                output_tokens.extend(cur_output_tokens)
                cur_output_tokens.append(EOS)
                while len(cur_output_tokens) < max_seq_len:
                    cur_output_tokens.append(PAD)
                output_seqs.append(cur_output_tokens)

        fr_vocab = text.vocab.Vocabulary(
            collections.Counter(input_tokens), reserved_tokens=[PAD, BOS, EOS])
        en_vocab = text.vocab.Vocabulary(
            collections.Counter(output_tokens), reserved_tokens=[PAD, BOS, EOS])
    return fr_vocab, en_vocab, input_seqs, output_seqs


# 创建训练数据集
input_vocab, output_vocab, input_seqs, output_seqs = read_data(max_seq_len)
X = nd.zeros((len(input_seqs), max_seq_len), ctx=ctx)
Y = nd.zeros((len(output_seqs), max_seq_len), ctx=ctx)
for i in range(len(input_seqs)):
    X[i] = nd.array(input_vocab.to_indices(input_seqs[i]), ctx=ctx)
    Y[i] = nd.array(output_vocab.to_indices(output_seqs[i]), ctx=ctx)
dataset = gluon.data.ArrayDataset(X, Y)


# 以下定义了基于GRU的编码器
class Encoder(Block):
    """编码器"""

    def __init__(self, input_dim, hidden_dim, num_layers, drop_prob, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        with self.name_scope():
            # input_dim is len(input_vocab)
            # hidden_dim is encoder_hidden_dim, 256
            # embedding.weight.shape is (input_dim, hidden_dim)
            self.embedding = nn.Embedding(input_dim, hidden_dim)
            self.dropout = nn.Dropout(drop_prob)
            # num_layers is 1
            self.rnn = rnn.GRU(
                hidden_dim, num_layers, dropout=drop_prob, input_size=hidden_dim)

    def forward(self, inputs, state):
        # inputs尺寸: (batch_size, max_seq_len)，emb尺寸: (max_seq_len, batch_size, 256)
        emb = self.embedding(inputs).swapaxes(0, 1)
        emb = self.dropout(emb)
        # output.shape is (max_seq_len, batch_size, 256)
        # state.shape is (layer_size, batch_size, hidden_dim)
        # we need the state of last GRU layer, state[-1]
        output, state = self.rnn(emb, state)
        return output, state

    def begin_state(self, *args, **kwargs):
        # begin_state(batch_size)
        return self.rnn.begin_state(*args, **kwargs)


# 以下定义了基于GRU的解码器
class Decoder(Block):
    """含注意力机制的解码器"""

    def __init__(self, hidden_dim, output_dim, num_layers, max_seq_len, drop_prob,
                 alignment_dim, encoder_hidden_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.encoder_hidden_dim = encoder_hidden_dim
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        with self.name_scope():
            # hidden_dim is decoder_hidden_dim
            # output_dim is len(output_vocab)
            self.embedding = nn.Embedding(output_dim, hidden_dim)
            self.dropout = nn.Dropout(drop_prob)

            # 注意力机制。
            self.attention = nn.Sequential()
            with self.attention.name_scope():
                # the layer output (*V, in_units) -> (*V, alignment_dim)
                # namely (*V, hidden_dim + encoder_hidden_dim) -> (*V, alignment_dim)
                self.attention.add(
                    # alignment_dim is 25
                    nn.Dense(
                        alignment_dim,
                        in_units=hidden_dim + encoder_hidden_dim,
                        activation="tanh",
                        flatten=False))

                # the layer output (*V, alignment_dim) -> (*V, 1)
                # activation = None
                self.attention.add(nn.Dense(1, in_units=alignment_dim, flatten=False))

            self.rnn = rnn.GRU(
                hidden_dim, num_layers, dropout=drop_prob, input_size=hidden_dim)

            # the layer output (*V, hidden_dim) -> (*V, output_dim)
            # activation = None
            self.out = nn.Dense(output_dim, in_units=hidden_dim, flatten=False)

            # the layer output (*V, hidden_dim + encoder_hidden_dim) -> (*V, hidden_dim)
            # activation = None
            self.rnn_concat_input = nn.Dense(
                hidden_dim, in_units=hidden_dim + encoder_hidden_dim, flatten=False)

    def forward(self, cur_input, state, encoder_outputs):
        # 当RNN为多层时，取最靠近输出层的单层隐含状态。
        # state.shape is [(1, batch_size, decoder_hidden_dim)]
        single_layer_state = [state[0][-1].expand_dims(0)]
        # encoder_outputs.shape is (max_seq_len, batch_size * encoder_hidden_dim)
        encoder_outputs = encoder_outputs.reshape((self.max_seq_len, -1,
                                                   self.encoder_hidden_dim))
        # single_layer_state尺寸: [(1, batch_size, decoder_hidden_dim)]
        # hidden_broadcast尺寸: (max_seq_len, batch_size, decoder_hidden_dim)
        hidden_broadcast = nd.broadcast_axis(
            single_layer_state[0], axis=0, size=self.max_seq_len)

        # encoder_outputs_and_hiddens尺寸:
        # (max_seq_len, batch_size, encoder_hidden_dim + decoder_hidden_dim)
        encoder_outputs_and_hiddens = nd.concat(encoder_outputs, hidden_broadcast, dim=2)

        # energy尺寸: (max_seq_len, batch_size, 1)
        energy = self.attention(encoder_outputs_and_hiddens)

        # batch_attention尺寸: (batch_size, 1, max_seq_len)
        batch_attention = nd.softmax(energy, axis=0).transpose((1, 2, 0))

        # batch_encoder_outputs尺寸: (batch_size, max_seq_len, encoder_hidden_dim)
        batch_encoder_outputs = encoder_outputs.swapaxes(0, 1)

        # decoder_context尺寸: (batch_size, 1, encoder_hidden_dim)
        decoder_context = nd.batch_dot(batch_attention, batch_encoder_outputs)

        # cur_input尺寸: (batch_size,)
        # input_and_context尺寸: (batch_size, 1, decoder_hidden_dim + encoder_hidden_dim )
        input_and_context = nd.concat(
            nd.expand_dims(self.embedding(cur_input), axis=1), decoder_context, dim=2)
        # concat_input尺寸: (1, batch_size, decoder_hidden_dim)
        concat_input = self.rnn_concat_input(input_and_context).reshape((1, -1, 0))
        concat_input = self.dropout(concat_input)

        # 当RNN为多层时，用单层隐含状态初始化各个层的隐含状态。
        state = [nd.broadcast_axis(single_layer_state[0], axis=0, size=self.num_layers)]

        # XXX 注意：state 是 [nd.NDArray]
        output, state = self.rnn(concat_input, state)
        output = self.dropout(output)
        output = self.out(output)
        output = nd.reshape(output, (-3, -1))
        # output尺寸: (batch_size * 1, output_dim)
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


# 为了初始化解码器的隐含状态，我们通过一层全连接网络来转化 编码器的 输出隐含状态。
class DecoderInitState(Block):
    """解码器隐含状态的初始化"""

    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, **kwargs):
        super(DecoderInitState, self).__init__(**kwargs)
        with self.name_scope():
            # (encoder_num_layers, batch_size, encoder_hidden_dim) -> 
            # (encoder_num_layers, batch_size, decoder_hidden_dim)
            self.dense = nn.Dense(
                decoder_hidden_dim,
                in_units=encoder_hidden_dim,
                activation="tanh",
                flatten=False)

    def forward(self, encoder_state):
        return [self.dense(encoder_state)]


# 我们定义translate函数来应用训练好的模型。
# 解码器的最初时刻输入来自BOS字符。
# 当任一时刻的输出为EOS字符时，输出序列即完成。
def translate(encoder, decoder, decoder_init_state, fr_ens, ctx, max_seq_len):
    for fr_en in fr_ens:
        print('Input :', fr_en[0])
        input_tokens = fr_en[0].split(' ') + [EOS]
        # 添加PAD符号使每个序列等长（长度为max_seq_len）。
        while len(input_tokens) < max_seq_len:
            input_tokens.append(PAD)
        inputs = nd.array(input_vocab.to_indices(input_tokens), ctx=ctx)
        encoder_state = encoder.begin_state(func=mx.nd.zeros, batch_size=1, ctx=ctx)
        encoder_outputs, encoder_state = encoder(inputs.expand_dims(0), encoder_state)
        encoder_outputs = encoder_outputs.flatten()
        # 解码器的第一个输入为BOS字符。
        decoder_input = nd.array([output_vocab.token_to_idx[BOS]], ctx=ctx)
        decoder_state = decoder_init_state(encoder_state[0])
        output_tokens = []

        for _ in range(max_output_len):
            decoder_output, decoder_state = decoder(decoder_input, decoder_state,
                                                    encoder_outputs)
            pred_i = int(decoder_output.argmax(axis=1).asnumpy()[0])
            # 当任一时刻的输出为EOS字符时，输出序列即完成。
            if pred_i == output_vocab.token_to_idx[EOS]:
                break
            else:
                output_tokens.append(output_vocab.idx_to_token[pred_i])
            decoder_input = nd.array([pred_i], ctx=ctx)

        print('Output:', ' '.join(output_tokens))
        print('Expect:', fr_en[1], '\n')


# 模型训练
def train(encoder, decoder, decoder_init_state, max_seq_len, ctx, eval_fr_ens):
    # 对于三个网络，分别初始化它们的模型参数并定义它们的优化器。
    encoder.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    decoder.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    decoder_init_state.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    encoder_optimizer = gluon.Trainer(encoder.collect_params(), 'adam',
                                      {'learning_rate': learning_rate})
    decoder_optimizer = gluon.Trainer(decoder.collect_params(), 'adam',
                                      {'learning_rate': learning_rate})
    decoder_init_state_optimizer = gluon.Trainer(decoder_init_state.collect_params(), 'adam',
                                                 {'learning_rate': learning_rate})

    softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    prev_time = datetime.datetime.now()
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

    total_loss = 0.0
    eos_id = output_vocab.token_to_idx[EOS]
    for epoch in range(1, epochs + 1):
        for x, y in data_iter:
            real_batch_size = x.shape[0]
            with autograd.record():
                loss = nd.array([0], ctx=ctx)
                valid_length = nd.array([0], ctx=ctx)
                encoder_state = encoder.begin_state(
                    func=mx.nd.zeros, batch_size=real_batch_size, ctx=ctx)
                encoder_outputs, encoder_state = encoder(x, encoder_state)

                # encoder_outputs尺寸:  (max_seq_len, batch_size * encoder_hidden_dim)
                encoder_outputs = encoder_outputs.flatten()
                # 解码器的第一个输入为BOS字符。
                decoder_input = nd.array(
                    [output_vocab.token_to_idx[BOS]] * real_batch_size, ctx=ctx)
                mask = nd.ones(shape=(real_batch_size,), ctx=ctx)

                # XXX 注意：encoder_state 是 [nd.NDArray]
                # encoder_state[0].shape is (1, batch_size, encoder_hidden_dim)
                # decoder_state is [(1, batch_size, decoder_hidden_dim)]
                decoder_state = decoder_init_state(encoder_state[0])
                for i in range(max_seq_len):
                    decoder_output, decoder_state = decoder(decoder_input, decoder_state,
                                                            encoder_outputs)
                    # 解码器使用当前时刻的预测结果作为下一时刻的输入。
                    decoder_input = decoder_output.argmax(axis=1)
                    valid_length = valid_length + mask.sum()
                    loss = loss + (mask * softmax_cross_entropy(decoder_output, y[:, i])).sum()
                    mask = mask * (y[:, i] != eos_id)
                loss = loss / valid_length
            loss.backward()
            encoder_optimizer.step(1)
            decoder_optimizer.step(1)
            decoder_init_state_optimizer.step(1)
            total_loss += loss.asscalar() / max_seq_len

        if epoch % epoch_period == 0 or epoch == 1:
            cur_time = datetime.datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = 'Time %02d:%02d:%02d' % (h, m, s)
            if epoch == 1:
                print_loss_avg = total_loss / len(data_iter)
            else:
                print_loss_avg = total_loss / epoch_period / len(data_iter)
            loss_str = 'Epoch %d, Loss %f, ' % (epoch, print_loss_avg)
            print(loss_str + time_str)
            if epoch != 1:
                total_loss = 0.0
            prev_time = cur_time

            translate(encoder, decoder, decoder_init_state, eval_fr_ens, ctx, max_seq_len)


# 实例化编码器、解码器和解码器初始隐含状态网络
encoder = Encoder(len(input_vocab), encoder_hidden_dim, encoder_num_layers, encoder_drop_prob)
decoder = Decoder(decoder_hidden_dim, len(output_vocab), decoder_num_layers, max_seq_len,
                  decoder_drop_prob, alignment_dim, encoder_hidden_dim)
decoder_init_state = DecoderInitState(encoder_hidden_dim, decoder_hidden_dim)

# 给定简单的法语和英语序列，我们可以观察模型的训练结果。
# 打印的结果中，Input、Output和Expect分别代表输入序列、输出序列和正确序列。
eval_fr_ens = [['elle est japonaise .', 'she is japanese .'],
               ['ils sont russes .', 'they are russian .'],
               ['ils regardent .', 'they are watching .']]
train(encoder, decoder, decoder_init_state, max_seq_len, ctx, eval_fr_ens)
