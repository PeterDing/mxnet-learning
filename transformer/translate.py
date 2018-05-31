from mxnet import nd
from data import make_src_mask, make_trg_mask

MAX_LEN = 20

def translate(net, src, trg_vocab, s_pad, t_bos, t_eos, t_pad):
    src_mask = make_src_mask(src, s_pad)

    trg_list = [t_bos]

    for _ in range(MAX_LEN):
        trg = nd.array([trg_vocab.to_indices(trg_list)])
        trg_mask = make_trg_mask(trg, t_pad)

        pred = net(src, trg, src_mask, trg_mask)
        out = pred.argmax(-1)
        next_idx = out[-1][0].asscalar()
        if next_idx == t_eos:
            break
        trg_list.append(int(next_idx))

    return trg_vocab.to_tokens(trg_list)
