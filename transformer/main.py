from mxnet import gluon, autograd, nd
import mxnet

from constants import BOS, EOS, PAD
from data import load_data, make_src_mask, make_trg_mask
from model import make_net, Generator
from train import ReduceLRScheduler, get_loss
import numpy as np
from translate import translate

ctx = mxnet.cpu()

# hyper params
epoch = 100
data_dir = '../data/iwslt16/de-en'
src_lang = 'de'
trg_lang = 'en'
limit = 10
batch_size = 1

# for net
num_layer = 6
model_dim = 512
h = 8
ff_dim = 2048
dropout = 0.1

# for adam
learning_rate = model_dim**-0.5
beta1 = 0.9
beta2 = 0.98
epsilon = 1e-9
warmup_steps = 30  # 4000

# for loss
smooth_alpha = 0.1

data_iter, src_vocab, trg_vocab = load_data(
    data_dir, src_lang, trg_lang, limit=limit, batch_size=1, shuffle=True)
s_pad = src_vocab.to_indices(PAD)
t_bos = trg_vocab.to_indices(BOS)
t_eos = trg_vocab.to_indices(EOS)
t_pad = trg_vocab.to_indices(PAD)
num_classes = len(trg_vocab)


xxx = ['Wir', 'werden', 'Ihnen', 'einige', 'Geschichten', 'über', 'das', 'Meer', 'in', 'Videoform', 'erzählen', '.']
yyy = "And we're going to tell you some stories from the sea here in video."
xxx_src = nd.array(src_vocab.to_indices(xxx)).expand_dims(0)


def get_trainer(params, ctx=None):
    optimizer = mxnet.optimizer.Adam(
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        lazy_update=True,
        lr_scheduler=ReduceLRScheduler(model_dim, warmup_steps))
    trainer = gluon.Trainer(params, optimizer)
    return trainer


def train(net):
    net.initialize(init=mxnet.init.Xavier(), ctx=ctx)

    net_trainer = get_trainer(net.collect_params(), ctx)

    for i in range(epoch):
        tloss = 0
        for (src, trg), trg_y in data_iter:
            src_mask = make_src_mask(src, s_pad)
            trg_mask = make_trg_mask(trg, t_pad)

            with autograd.record():
                pred = net(src, trg, src_mask, trg_mask)
                loss = get_loss(pred, trg_y, num_classes, t_pad, smooth_alpha=smooth_alpha)
            loss.backward()
            net_trainer.step(1)
            tloss += loss.asscalar()
        print('epoch: {}, loss: {}'.format(i, tloss))
        print(' '.join(translate(net, xxx_src, trg_vocab, s_pad, t_bos, t_eos, t_pad)))
        print(yyy)


def main():
    net = make_net(
        len(src_vocab),
        len(trg_vocab),
        model_dim=model_dim,
        num_layer=num_layer,
        ff_dim=ff_dim,
        h=h,
        dropout=dropout)

    train(net)


if __name__ == '__main__':
    main()
