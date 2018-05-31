from mxnet import nd
from mxnet.gluon import loss as gloss

# smoothed cross_entropy
# https://github.com/awslabs/sockeye/blob/master/sockeye/loss.py#L153
def get_loss(pred, label, num_classes, trg_pad, smooth_alpha=0.1):
    pred = nd.maximum(pred, 1e-10)
    logprob = nd.log_softmax(pred)

    # cross entropy
    ce = -nd.pick(logprob, label)

    pre_class_gain = smooth_alpha / (num_classes - 1)

    # loss = (1 - smooth_alpha - pre_class_gain) * ce - pre_class_gain * sum(logprob)
    loss = (1 - smooth_alpha - pre_class_gain) * ce - nd.sum(
        pre_class_gain * logprob, axis=-1, keepdims=False)

    mask = (label != trg_pad).astype(dtype=pred.dtype)
    loss *= mask

    loss = nd.sum(loss) / mask.sum()

    return loss
