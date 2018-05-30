# TODO, mask defination

import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn


def register_children(block, children):
    assert isinstance(children, (list, tuple))
    for child in children:
        block.register_child(child, name=child.name)


class ResNorm(nn.Block):

    def __init__(self, dropout):
        super().__init__()

        with self.name_scope():
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class FeedForward(nn.Block):

    def __init__(self, hidden_dim, output_dim, dropout=0.1):
        super().__init__()

        with self.name_scope():
            self.denses = [
                nn.Dense(hidden_dim, activation='relu', flatten=False),
                nn.Dense(output_dim, flatten=False)
            ]
            register_children(self, self.denses)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.denses[1](self.dropout(self.denses[0](x)))


class PositionalEmbedding(nn.Block):

    def __init__(self, vocab_size, model_dim, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim

        with self.name_scope():
            self.embedding = nn.Embedding(vocab_size, model_dim)
            self.dropout = nn.Dropout(dropout)

    @staticmethod
    def positional(x):
        batch_size, length, model_dim = x.shape
        # (length, 1)
        pos = nd.arange(length).expand_dims(1)

        # (1, model_dim/2),  10000^(2i/model_dim)
        div = nd.power(10000, nd.arange(model_dim / 2) * 2 / model_dim)

        out = nd.zeros((length, model_dim))

        out[:, 0::2] = nd.sin(pos / div)
        out[:, 1::2] = nd.cos(pos / div)

        return nd.broadcast_axis(out.expand_dims(0), axis=0, size=batch_size)

    def forward(self, x):
        print('----', x)
        x = self.embedding(x) * nd.sqrt(nd.array([self.model_dim]))
        return self.dropout(x + self.positional(x))


def dot_attention(query, key, value, mask, dropout=0.0):
    # query: (batch_size, h, length_q, model_dim/h)
    # key:   (batch_size, h, length_k, model_dim/h)
    # value: (batch_size, h, length_k, model_dim/h)

    query_shape = query.shape
    query = query.reshape(-3, -2)
    key = key.reshape(-3, -2)
    value = value.reshape(-3, -2)

    # matmul, t: (batch_size*h, length_q, length_k)
    t = nd.batch_dot(query, key.swapaxes(1, 2)) / nd.sqrt(nd.array([query.shape[-1]]))

    # masked
    t = t * mask

    # softmax
    t = nd.softmax(t, axis=-1)
    if dropout > 0.0:
        t = nd.dropout(t, p=dropout)

    return nd.batch_dot(t, value).reshape(query_shape)


class MultiHeadAttention(nn.Block):

    def __init__(self, h=8, model_dim=512):
        super().__init__()
        self.h = h
        self.model_dim = model_dim

        with self.name_scope():
            self.denses = [nn.Dense(model_dim, flatten=False) for _ in range(4)]
            register_children(self, self.denses)

    def forward(self, query, key, value, mask):
        # query: (batch_size, length_q, model_dim)
        # key:   (batch_size, length_k, model_dim)
        # value: (batch_size, length_v, model_dim)
        query_shape = query.shape

        # query: (batch_size, h, length_q, model_dim/h)
        # key:   (batch_size, h, length_k, model_dim/h)
        # value: (batch_size, h, length_v, model_dim/h)
        query = self.denses[0](query).reshape(0, 0, self.h, -1).swapaxes(1, 2)
        key = self.denses[1](key).reshape(0, 0, self.h, -1).swapaxes(1, 2)
        value = self.denses[2](value).reshape(0, 0, self.h, -1).swapaxes(1, 2)

        # h - attention
        out = dot_attention(query, key, value, mask)

        # concat
        out = out.reshape(query_shape)
        return self.denses[3](out)


class EncoderDecoder(nn.Block):

    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 num_layer=6,
                 model_dim=512,
                 ff_dim=2048,
                 h=8,
                 dropout=0.1):

        super().__init__()
        self.num_layer = num_layer
        self.model_dim = model_dim
        self.dropout = dropout
        self.h = h

        with self.name_scope():
            self.encoder = Encoder(
                src_vocab_size,
                num_layer=num_layer,
                model_dim=model_dim,
                ff_dim=ff_dim,
                dropout=dropout)

            self.decoder = Decoder(
                trg_vocab_size,
                num_layer=num_layer,
                model_dim=model_dim,
                ff_dim=ff_dim,
                dropout=dropout)

    def forward(self, src, trg, src_mask, trg_mask):
        memory = self.encoder(src, src_mask)
        return self.decoder(memory, trg, src_mask, trg_mask)


class Encoder(nn.Block):

    def __init__(self, vocab_size, num_layer=6, model_dim=512, ff_dim=2048, h=8, dropout=0.1):
        super().__init__()
        self.num_layer = num_layer
        self.model_dim = model_dim
        self.dropout = dropout
        self.h = h

        with self.name_scope():
            self.encoder_layers = [
                EncoderLayer(model_dim=model_dim, ff_dim=ff_dim, h=h, dropout=dropout)
                for _ in range(num_layer)
            ]
            register_children(self, self.encoder_layers)
            self.norm = nn.LayerNorm()
            self.embedding_position = PositionalEmbedding(
                vocab_size, model_dim, dropout=dropout)

    def forward(self, src, src_mask):
        print('src', src)
        x = self.embedding_position(src)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.norm(x)


class EncoderLayer(nn.Block):

    def __init__(self, model_dim=512, ff_dim=2048, h=8, dropout=0.1):
        super().__init__()

        with self.name_scope():
            self.self_attention = MultiHeadAttention(h=h, model_dim=model_dim)
            self.ff = FeedForward(ff_dim, model_dim, dropout=dropout)
            #  self.dropout = nn.Dropout(dropout)
            #  self.norms = [nn.LayerNorm() for _ in range(2)]
            self.res_norms = [ResNorm(dropout) for _ in range(2)]
            register_children(self, self.res_norms)

    def forward(self, src, src_mask):
        #  src = src + self.dropout(
        #  (lambda x: self.self_attention(x, x, x, src_mask))(self.norms[0](src)))
        #  src = src + self.dropout(self.ff(self.norms[1](src)))
        #  return src

        src = self.res_norms[0](src, lambda x: self.self_attention(x, x, x, src_mask))
        return self.res_norms[1](src, self.ff)


class Generator(nn.Block):

    def __init__(self, vocab_size):
        self.dense = nn.Dense(vocab_size, flatten=False)

    def forward(self, pred):
        return self.dense(pred)


class Decoder(nn.Block):

    def __init__(self, vocab_size, num_layer=6, model_dim=512, ff_dim=2048, h=8, dropout=0.1):
        super().__init__()
        self.num_layer = num_layer
        self.model_dim = model_dim
        self.dropout = dropout
        self.h = h

        with self.name_scope():
            self.decoder_layers = [
                DecoderLayer(model_dim=model_dim, ff_dim=ff_dim, h=h, dropout=dropout)
                for _ in range(num_layer)
            ]
            register_children(self, self.decoder_layers)
            self.norm = nn.LayerNorm()
            self.embedding_position = PositionalEmbedding(
                vocab_size, model_dim, dropout=dropout)

    def forward(self, memory, trg, memory_mask, trg_mask):
        trg = self.embedding_position(trg)
        for layer in self.decoder_layers:
            trg = layer(memory, trg, memory_mask, trg_mask)
        return self.norm(trg)


class DecoderLayer(nn.Block):

    def __init__(self, model_dim=512, ff_dim=2048, h=8, dropout=0.1):
        super().__init__()
        self.h = h

        with self.name_scope():
            self.self_attention_masked = MultiHeadAttention(h=h, model_dim=model_dim)
            self.self_attention = MultiHeadAttention(h=h, model_dim=model_dim)
            self.ff = FeedForward(ff_dim, model_dim, dropout=dropout)
            #  self.dropout = nn.Dropout(dropout)
            #  self.norms = [nn.LayerNorm() for _ in range(3)]
            #  register_children(self, self.norms)
            self.res_norms = [ResNorm(dropout) for _ in range(3)]
            register_children(self, self.res_norms)

    def forward(self, memory, trg, memory_mask, trg_mask):
        # layer(query, key, value)
        #  trg = trg + self.dropout(
        #  (lambda x: self.self_attention_masked(x, x, x, trg_mask))(self.norms[0](trg)))
        #  trg = trg + self.dropout(
        #  (lambda x: self.self_attention_masked(x, memory, memory, memory_mask))(
        #  self.norms[1](trg)))
        #  trg = trg + self.dropout(self.ff(self.norms[2](trg)))
        #  return trg

        trg = self.res_norms[0](trg, lambda x: self.self_attention_masked(x, x, x, trg_mask))
        trg = self.res_norms[1](
            trg, lambda x: self.self_attention_masked(x, memory, memory, memory_mask))
        return self.res_norms[2](trg, self.ff)


def make_net(src_vocab_size,
             trg_vocab_size,
             num_layer=6,
             model_dim=512,
             ff_dim=2048,
             h=8,
             dropout=0.1):

    net = EncoderDecoder(
        src_vocab_size,
        trg_vocab_size,
        num_layer=num_layer,
        model_dim=model_dim,
        ff_dim=ff_dim,
        h=h,
        dropout=dropout)

    return net
