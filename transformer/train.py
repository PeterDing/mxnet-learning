from mxnet import nd
from mxnet.gluon import loss as gloss
from mxnet import lr_scheduler


# with Label smoothing
# https://arxiv.org/abs/1512.00567
def get_loss(pred, label, trg_vocab_size, trg_pad, epsilon=0.1):
    labelprob = nd.one_hot(label, trg_vocab_size)

    # Label smoothing
    smoothed_labelprob = (1 - epsilon) * labelprob + epsilon / trg_vocab_size

    logprob = nd.log_softmax(pred)

    loss = -nd.sum(logprob * smoothed_labelprob, axis=-1, keepdims=False)

    # mask PAD
    mask = label != trg_pad
    loss = loss * mask

    # batch_axis = 0
    loss = nd.mean(loss, axis=0, exclude=True)

    return loss


# smoothed cross_entropy
# https://github.com/awslabs/sockeye/blob/master/sockeye/loss.py#L153
def get_smoothed_loss(pred, label, num_classes, trg_pad, smooth_alpha=0.1):
    pred = nd.maximum(pred, 1e-10)
    logprob = nd.log_softmax(pred)

    # cross entropy
    ce = -nd.pick(logprob, label)

    pre_class_gain = smooth_alpha / (num_classes - 1)

    # loss = (1 - smooth_alpha - pre_class_gain) * ce - pre_class_gain * sum(logprob)
    loss = (1 - smooth_alpha - pre_class_gain) * ce - nd.sum(
        pre_class_gain * logprob, axis=-1, keepdims=False)

    mask = label != trg_pad
    loss = loss * mask

    loss = nd.sum(loss) / mask.sum()

    return loss


class ReduceLRScheduler(lr_scheduler.LRScheduler):

    def __init__(self, model_dim, warmup_steps):
        self.model_dim = model_dim
        self.warmup_steps = warmup_steps
        self._a = model_dim**-0.5
        self._b = self.warmup_steps**-1.5

    def __call__(self, num_update):
        # lr = model_dim^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
        if num_update < self.warmup_steps:
            lr = self._a * num_update * self._b
        else:
            lr = self._a * num_update**-0.5
        return lr
