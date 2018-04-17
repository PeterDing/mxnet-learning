import mxnet as mx
from mxnet import nd
from mxnet import autograd as autograd
from mxnet import gluon
from utils import SGD, accuracy, evaluate_accuracy, load_data_fashion_mnist

# 输入输出数据格式是 batch x channel x height x width，这里batch和channel都是1
# 权重格式是 output_channels x in_channels x height x width，这里input_filter和output_filter都是1。

# {{
# convolution
# nd.Convolution(data=None, weight=None, bias=None, kernel=_Null, stride=_Null, dilate=_Null, pad=_Null, num_filter=_Null, num_group=_Null, workspace=_Null, no_bias=_Null, cudnn_tune=_Null, cudnn_off=_Null, layout=_Null, out=None, name=None, **kwargs)
# data.shape = (batch, in_channels, *V)   # V 是N-D空间纬度
#                                         # 比如 4-D 的 V 是 a, b, c, d
# weight.shape = (output_channels, in_channels, *V)
# bias.shape = (output_channels,)
# out.shape = (batch, output_channels, *VV)   # VV 是 weight 扫过 data 后的纬度
# 这里 num_filter = output_channels

# 2-D convolution 计算
# out[i,j,:,:] = \sum_{0}^{n} \sum_{0}^{m} data[i+n,j+m,:,:] \star weight[n,m,:,:] + bias  # \star 是按元素乘
# 多余多个通道，需要于对应的 weight 通道计算完卷积后，把所得值加和，然后再加 bias。

# e.g.
# In [212]: w = nd.arange(48).reshape((3,2,2,2,2))
#      ...: b = nd.zeros(w.shape[0])
#      ...: data = nd.arange(54).reshape((1,2,3,3,3))
#      ...: out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0])
#      ...:
#      ...: print('input:', data, '\n\nweight:', w, '\n\nbias:', b, '\n\noutput:', out)
# }}

try:
    ctx = mx.gpu()
    _ = nd.zeros((1, ), ctx=ctx)
except:
    ctx = mx.cpu()

batch_size = 256
train_data, test_data = load_data_fashion_mnist(batch_size)

#########################################################################
# params build

weight_scale = .01

# 下面“=”都指 shape
# data1 = (256, 1, 28, 28)

# w1 = (20, 1, 5, 5)
# b1 = (20, )
# out1 = (256, 20, 24, 24)  # 28 - 5 + 1
# out1 (after pooling) = (256, 20, 12, 12)   # stride = (2, 2), 24 / 2 = 12
W1 = nd.random_normal(shape=(20, 1, 5, 5), scale=weight_scale, ctx=ctx)
b1 = nd.zeros(W1.shape[0], ctx=ctx)

# data2 = out1
# w2 = (50, 20, 3, 3)
# b2 = (50, )
# out2 = (256, 50, 10, 10)  # 12 - 3 + 1
# out2 (after pooling) = (256, 50, 5, 5)   # stride = (2, 2), 10 / 2 = 5
# out2 (after flatten) = (256, 1250)  # 50 * 5 * 5
W2 = nd.random_normal(shape=(50, 20, 3, 3), scale=weight_scale, ctx=ctx)
b2 = nd.zeros(W2.shape[0], ctx=ctx)

# data3 = out2
# w3 = (1250, 128)
# out3 = (256, 128)  # (256, 1250) x (1250, 128)
W3 = nd.random_normal(shape=(1250, 128), scale=weight_scale, ctx=ctx)
b3 = nd.zeros(W3.shape[1], ctx=ctx)

# data4 = out4
# w4 = (128, 10)
# out4 = (256, 10)  # (256, 128) x (128, 10)
W4 = nd.random_normal(shape=(W3.shape[1], 10), scale=weight_scale, ctx=ctx)
b4 = nd.zeros(W4.shape[1], ctx=ctx)

params = [W1, b1, W2, b2, W3, b3, W4, b4]
for param in params:
    param.attach_grad()


# bulid model
# 卷积模块通常是“卷积层-激活层-池化层”。然后转成2D矩阵输出给后面的全连接层。
def net(X, verbose=False):
    X = X.as_in_context(W1.context)
    # 第一层卷积
    h1_conv = nd.Convolution(
        data=X,
        weight=W1,
        bias=b1,
        kernel=W1.shape[2:],
        num_filter=W1.shape[0])
    h1_activation = nd.relu(h1_conv)
    h1 = nd.Pooling(
        data=h1_activation, pool_type="max", kernel=(2, 2), stride=(2, 2))
    # 第二层卷积
    h2_conv = nd.Convolution(
        data=h1,
        weight=W2,
        bias=b2,
        kernel=W2.shape[2:],
        num_filter=W2.shape[0])
    h2_activation = nd.relu(h2_conv)
    h2 = nd.Pooling(
        data=h2_activation, pool_type="max", kernel=(2, 2), stride=(2, 2))
    h2 = nd.flatten(h2)
    # 第一层全连接
    h3_linear = nd.dot(h2, W3) + b3
    h3 = nd.relu(h3_linear)
    # 第二层全连接
    h4_linear = nd.dot(h3, W4) + b4
    if verbose:
        print('1st conv block:', h1.shape)
        print('2nd conv block:', h2.shape)
        print('1st dense:', h3.shape)
        print('2nd dense:', h4_linear.shape)
        print('output:', h4_linear)
    return h4_linear


softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

learning_rate = .2


def train():
    for epoch in range(5):
        train_loss = 0.
        train_acc = 0.
        for data, label in train_data:
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss = softmax_cross_entropy(output, label)
            loss.backward()
            SGD(params, learning_rate / batch_size)

            train_loss += nd.mean(loss).asscalar()
            train_acc += accuracy(output, label)

        test_acc = evaluate_accuracy(test_data, net, ctx)
        print("Epoch %d. Loss: %f, Train acc %f, Test acc %f" %
              (epoch, train_loss / len(train_data),
               train_acc / len(train_data), test_acc))


def main():
    train()


if __name__ == '__main__':
    main()
