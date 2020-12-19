import gzip
import pickle
import numpy as np

# 设置输出值位数：全部输出
np.set_printoptions(threshold=np.inf)

"""
参考：https://www.cnblogs.com/qxcheng/p/11729773.html
"""


# 独热编码转换
def onehot(targets, num):
    result = np.zeros((num, 10))
    for i in range(num):
        result[i][targets[i]] = 1
    return result


# 生成特征图并返回
def img2col(img, ksize, stride):
    # ksize: 卷积核大小
    # stride: 步长
    w, h, cnum = img.shape  # 图像宽、高、通道数
    fsize = (w - ksize) // stride + 1   # 特征图大小
    images = np.zeros((fsize * fsize, ksize * ksize * cnum))
    idx = 0
    for i in range(fsize):
        for j in range(fsize):
            images[idx] = img[i * stride:i * stride + ksize,
                              j * stride:j * stride + ksize, :].flatten()
            idx += 1
    return images


# 卷积层
class Conv(object):
    def __init__(self, kernel_shape, stride=1, pad=0):
        width, height, in_channel, out_channel = kernel_shape
        self.stride = stride
        self.pad = pad
        scale = np.sqrt(3 * in_channel * width * height / out_channel)
        self.k = np.random.standard_normal(kernel_shape) / scale
        self.b = np.random.standard_normal(out_channel) / scale
        self.k_gradient = np.zeros(kernel_shape)
        self.b_gradient = np.zeros(out_channel)

    def forward(self, x):
        self.x = x
        if self.pad != 0:
            self.x = np.pad(self.x, ((0, 0), (self.pad, self.pad),
                                     (self.pad, self.pad), (0, 0)), 'constant')
        bx, wx, hx, cx = self.x.shape   # batchsize，width，height，channelnum
        wk, hk, ck, nk = self.k.shape             # kernel的宽、高、通道数、个数:5,5,1,6
        feature_w = (wx - wk) // self.stride + 1  # 特征图尺寸
        feature = np.zeros((bx, feature_w, feature_w, nk))  # 10,24,24,6

        self.image_col = []  # 10,576,25，列表（一维向量）形式(采用append函数生成)
        kernel = self.k.reshape(-1, nk)  # (25, 6)
        for i in range(bx):
            image_col = img2col(self.x[i], wk, self.stride)  # (24x24, 25)
            feature[i] = (image_col @ kernel + self.b
                          ).reshape(feature_w, feature_w, nk)  # 24，24，6
            self.image_col.append(image_col)
        # print(np.array(self.image_col).shape);exit()
        return feature  # 10,24,24,6

    def backward(self, delta, learning_rate):
        bx, wx, hx, cx = self.x.shape  # batch,14,14,inchannel
        wk, hk, ck, nk = self.k.shape  # 5,5,inChannel,outChannel
        bd, wd, hd, cd = delta.shape  # batch,10,10,outChannel

        # 计算self.k_gradient,self.b_gradient
        delta_col = delta.reshape(bd, -1, cd)
        for i in range(bx):
            self.k_gradient += (self.image_col[i].T @ delta_col[i]
                                ).reshape(self.k.shape)
        self.k_gradient /= bx
        self.b_gradient += np.sum(delta_col, axis=(0, 1))
        self.b_gradient /= bx

        # 计算delta_backward
        delta_backward = np.zeros(self.x.shape)
        k_180 = np.rot90(self.k, 2, (0, 1))      # numpy矩阵旋转180度
        k_180 = k_180.swapaxes(2, 3)
        k_180_col = k_180.reshape(-1, ck)

        if hd - hk + 1 != hx:
            pad = (hx - hd + hk - 1) // 2
            pad_delta = np.pad(
                delta, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        else:
            pad_delta = delta

        for i in range(bx):
            pad_delta_col = img2col(pad_delta[i], wk, self.stride)
            delta_backward[i] = (
                pad_delta_col @ k_180_col).reshape(wx, hx, ck)

        # 反向传播
        self.k -= self.k_gradient * learning_rate
        self.b -= self.b_gradient * learning_rate

        return delta_backward


# pool
class Pool(object):
    def forward(self, x):
        b, w, h, c = x.shape
        new_w, new_h = w // 2, h // 2
        feature = np.zeros((b, new_w, new_h, c))

        # 记录最大池化时最大值的位置信息用于反向传播
        self.feature_mask = np.zeros((b, w, h, c))
        for bi in range(b):
            for ci in range(c):
                for i in range(new_w):
                    for j in range(new_h):
                        feature[bi, i, j, ci] = np.max(
                            x[bi, i * 2:i * 2 + 2, j * 2:j * 2 + 2, ci])
                        index = np.argmax(
                            x[bi, i * 2:i * 2 + 2, j * 2:j * 2 + 2, ci])
                        self.feature_mask[bi, i * 2 + index // 2,
                                          j * 2 + index % 2, ci] = 1
        return feature  # 返回10，12，12，6的数组

    def backward(self, delta):
        # 向两个方向扩展维度，并与存储最大值位置的矩阵做Hadamard积
        return np.repeat(
            delta.repeat(2, axis=1), 2, axis=2) * self.feature_mask


# Relu
class Relu(object):
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, delta):
        delta[self.x < 0] = 0
        return delta


# 全连接层
class Linear(object):
    def __init__(self, inChannel, outChannel):
        scale = np.sqrt(inChannel / 2)
        self.W = np.random.randn(inChannel, outChannel) / scale  # 256,10
        self.b = np.random.standard_normal(outChannel) / scale  # 10,
        self.W_gradient = np.zeros((inChannel, outChannel))
        self.b_gradient = np.zeros(outChannel)

    def forward(self, x):
        self.x = x
        x_forward = self.x @ self.W + self.b
        return x_forward

    def backward(self, delta, learning_rate):
        # 梯度计算
        batch_size = self.x.shape[0]
        self.W_gradient = self.x.T @ delta / batch_size  # bxin bxout
        self.b_gradient = np.sum(delta, axis=0) / batch_size
        delta_backward = delta @ self.W.T                # bxout inxout
        # 反向传播
        self.W -= self.W_gradient * learning_rate
        self.b -= self.b_gradient * learning_rate

        return delta_backward


# Softmax
class Softmax(object):
    def cal_loss(self, predict, label):
        batchsize, classes = predict.shape  # 10，10
        self.predict(predict)
        loss = 0
        delta = np.zeros(predict.shape)
        for i in range(batchsize):
            delta[i] = self.softmax[i] - label[i]
            loss -= np.sum(np.log(self.softmax[i]) * label[i])
        loss /= batchsize
        return loss, delta

    def predict(self, predict):
        batchsize, classes = predict.shape  # 10，10
        self.softmax = np.zeros(predict.shape)

        for i in range(batchsize):
            predict_tmp = np.exp(predict[i] - np.max(predict[i]))
            # 计算batch中每一个元素的softmax概率并返回
            self.softmax[i] = predict_tmp / np.sum(predict_tmp)

        return self.softmax  # 10,10


def train(train_images, train_labels, ksize=3, batch_size=5, lr=.005, epoch=3):
    # 训练
    conv = Conv(kernel_shape=(ksize, ksize, 1, 8))  # 26x26x8
    pool = Pool()                         # 13x13x8
    pool_size = (28 - ksize + 1) // 2
    nn = Linear(pool_size * pool_size * 8, 10)
    softmax = Softmax()

    for ep in range(epoch):
        for i in range(0, len(train_images), batch_size):
            X = train_images[i:i + batch_size]
            Y = train_labels[i:i + batch_size]

            predict = conv.forward(X)  # 10,13,13,8
            predict = pool.forward(predict)  # 10,13,13,8
            predict = nn.forward(predict.reshape(batch_size, -1))

            loss, delta = softmax.cal_loss(predict, Y)

            delta = nn.backward(delta, lr)
            delta = delta.reshape(batch_size, pool_size, pool_size, 8)
            delta = pool.backward(delta)
            conv.backward(delta, lr)

            print("Epoch-{}-{:05d}".format(str(ep + 1), i + batch_size),
                  ":", "loss:{:.4f}".format(loss))
        lr *= .95**(ep + 1)

        np.savez("params.npz", k=conv.k, b=conv.b, W=nn.W, nb=nn.b)


def eval(test_images, test_labels, ksize=3, batch_size=5):
    r = np.load("params.npz")

    conv = Conv(kernel_shape=(ksize, ksize, 1, 8))  # 26x26x8
    pool = Pool()  # 13x13x8
    pool_size = (28 - ksize + 1) // 2
    nn = Linear(pool_size * pool_size * 8, 10)
    softmax = Softmax()

    conv.k = r["k"]
    conv.b = r["b"]
    nn.W = r["W"]
    nn.b = r["nb"]

    num = 0
    for i in range(len(test_images)):
        X = test_images[i]
        X = X[np.newaxis, :]
        Y = test_labels[i]

        predict = conv.forward(X)
        predict = pool.forward(predict)
        predict = nn.forward(predict.reshape(1, -1))

        predict = softmax.predict(predict)

        if np.argmax(predict) == Y:
            num += 1

    print("TEST-ACC: ", num / len(test_images) * 100, "%")


if __name__ == '__main__':
    # 导入数据（image及label）
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        training, validation, test = pickle.load(f, encoding='bytes')

    # 训练数据(total:50000)
    tr = 3000
    shuffle1 = np.random.permutation(tr)
    train_images = training[0][:tr].reshape(tr, 28, 28, 1)[shuffle1]
    # 标签one-hot处理 (60000, 10)
    train_labels = onehot(training[1][:tr], tr)[shuffle1]

    # 测试数据(total:10000)
    te = 1000
    shuffle2 = np.random.permutation(te)
    test_images = test[0][:te].reshape(te, 28, 28, 1)[shuffle2]
    test_labels = test[1][:te][shuffle2]

    print("训练模型")
    train(train_images, train_labels)

    print("测试模型")
    eval(test_images, test_labels)
