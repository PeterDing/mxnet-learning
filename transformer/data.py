# data IWSLT'16 (de-en) https://wit3.fbk.eu/archive/2016-01//texts/de/en/de-en.tgz

import collections
import os

import numpy as np
from mxnet import nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata

from constants import BOS, EOS, PAD

MAX_DATA_LEN = 1000000


def sequence_mask(size):
    mask = nd.array(np.tril(np.ones((1, size, size)), k=0))
    return mask


def make_src_mask(src, pad):
    return src != pad


def make_trg_mask(trg, pad):
    trg_mask = trg != pad
    smask = sequence_mask(trg.shape[-1])
    return trg_mask * smask


def make_vocab(vocab_list):
    vocab = text.vocab.Vocabulary(
        counter=collections.Counter(vocab_list),
        unknown_token='<unk>',
        reserved_tokens=[BOS, EOS, PAD])
    return vocab


def _transform(src, trg, s_bos, s_eos, s_pad, t_bos, t_eos, t_pad):
    src = list(src)
    trg = list(trg)

    # add eos to src end
    src.append(s_eos)

    # add bos to trg begin
    trg.insert(0, t_bos)

    # add eos to try_y
    trg_y = list(trg[1:])
    trg_y.append(t_eos)

    x = (nd.array(src), nd.array(trg))
    y = nd.array(trg_y)

    return x, y


def _load_data(path, limit=None):
    data = []
    vocab_set = set()
    with open(path) as f:
        for _ in range(min(limit or MAX_DATA_LEN, MAX_DATA_LEN)):
            line = f.readline().strip()
            if not line:
                break

            if not line.startswith('<'):
                sentence = line.replace(',', ' ,').replace('.', ' .').split()
                sentence = [w.strip() for w in sentence if w.strip()]
                data.append(sentence)
                vocab_set.update(sentence)
    vocab_list = sorted(vocab_set)
    vocab = make_vocab(vocab_list)
    data = [vocab.to_indices(tokens) for tokens in data]
    return vocab, data


def make_dataset(src_data, trg_data, src_vocab, trg_vocab):
    s_bos = src_vocab.to_indices(BOS)
    s_eos = src_vocab.to_indices(EOS)
    s_pad = src_vocab.to_indices(PAD)
    t_bos = src_vocab.to_indices(BOS)
    t_eos = src_vocab.to_indices(EOS)
    t_pad = src_vocab.to_indices(PAD)

    t = gdata.ArrayDataset(src_data, trg_data)
    dataset = t.transform(
        lambda x, y: _transform(x, y, s_bos, s_eos, s_pad, t_bos, t_eos, t_pad), lazy=True)
    return dataset


def load_data(root_dir, src_lang, trg_lang, limit=None, batch_size=1, shuffle=False):
    src_vocab, src_data = _load_data(
        os.path.join(root_dir, 'train.tags.{0}-{1}.{0}'.format(src_lang, trg_lang)),
        limit=limit)
    trg_vocab, trg_data = _load_data(
        os.path.join(root_dir, 'train.tags.{0}-{1}.{1}'.format(src_lang, trg_lang)),
        limit=limit)

    dataset = make_dataset(src_data, trg_data, src_vocab, trg_vocab)
    data_iter = gdata.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_iter, src_vocab, trg_vocab
