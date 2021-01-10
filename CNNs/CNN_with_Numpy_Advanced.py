import gzip
import pickle
import numpy as np

# 设置输出值位数：全部输出
np.set_printoptions(threshold=np.inf)

"""
参考：
https://zhuanlan.zhihu.com/p/102119808
https://www.cnblogs.com/qxcheng/p/11729773.html

===================模型基本结构===================
卷积层->最大值池化层->全连接层->Softmax层
=================================================
=====================参数选择=====================
batch size: 3
learning rate: 0.005
kernel size: 3x3
maxpool size: 2x2
kernel number(chadenseels): 8
padding: 0
stride(kernel): 1
stride(maxpool): 2
epoch: 3
=================================================
================数据维度变化(batch)===============
输入数据->(3,28,28,1)->卷积处理->(3,26,26,8)->
池化处理->(3,13,13,8)->(3,1352)->全连接层->
(3,10)->Softmax层(10,)->输出结果
=================================================
"""


# 独热编码转换
def onehot(targets, num):
    result = np.zeros((num, 10))
    for i in range(num):
        result[i][targets[i]] = 1
    return result


# 卷积层
class Conv(object):
    def __init__(self, kshape, stride=1, pad=0):
        width, height, cnum, knum = kshape
        self.stride = stride
        self.pad = pad
        scale = np.sqrt(3 * cnum * width * height / knum)

        # 初始化权重和偏置
        self.k = np.random.standard_normal(kshape) / scale
        self.b = np.random.standard_normal(knum) / scale

        # 初始化梯度
        self.k_gradient = np.zeros(kshape)
        self.b_gradient = np.zeros(knum)

    # 生成小图像矩阵
    def img2col(self, img, ksize, stride):
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

    def forward(self, x):
        self.x = x
        if self.pad != 0:
            self.x = np.pad(self.x,
                            ((0, 0), (self.pad, self.pad),
                             (self.pad, self.pad), (0, 0)),
                            'constant', constant_values=0)
        bsize, wx, hx, cx = self.x.shape  # 3,28,28,1
        # kernel的宽、高、输入通道数(图像通道数)、输出通道数:
        wk, hk, ck, nk = self.k.shape  # 3,3,1,8
        fsize = (wx - wk) // self.stride + 1  # 特征图大小
        fimgs = np.zeros((bsize, fsize, fsize, nk))  # 3,26,26,8

        self.image_col = []  # 3,676,9,列表(一维向量)形式(采用append函数生成)
        kernel = self.k.reshape(-1, nk)  # (3x3, 8)
        for i in range(bsize):
            image_col = self.img2col(self.x[i],
                                     wk, self.stride)  # (26x26, 3x3)
            fimgs[i] = (image_col @ kernel + self.b
                        ).reshape(fsize, fsize, nk)  # 26,26,8
            # 储存minibatch数据供反向传播使用
            self.image_col.append(image_col)
        return fimgs  # 3,26,26,8

    def backward(self, delta, lr):
        bsize, wx, hx, cx = self.x.shape  # 3,28,28,1
        wk, hk, ck, nk = self.k.shape  # 3,3,1,8
        bd, wd, hd, cd = delta.shape  # 池化层输出的梯度:3,26,26,8
        # 计算卷积核(权重)和偏置的梯度
        delta_col = delta.reshape(bd, -1, cd)  # 3,676,8

        for i in range(bsize):
            self.k_gradient += (self.image_col[i].T @ delta_col[i]
                                ).reshape(self.k.shape)  # 9,8->3,3,1,8

        self.k_gradient /= bsize
        self.b_gradient += np.sum(delta_col, axis=(0, 1))  # 8,
        self.b_gradient /= bsize

        # 计算前向传播的误差梯度
        delta_backward = np.zeros(self.x.shape)  # 3,28,28,1
        # 将卷积核矩阵旋转180°(上下颠倒，左右翻转)
        k_180 = np.rot90(self.k, 2, (0, 1))  # 3,3,1,8
        # 交换第3,4轴并展平,便于下面矩阵乘积运算
        k_180 = k_180.swapaxes(2, 3)  # 3,3,8,1
        k_180_col = k_180.reshape(-1, ck)  # 72,1

        # 若池化层输出的特征图大小(26)-卷积核大小(3)+1 != 原图像大小(28)
        # 则需要在转置卷积时对输出(误差矩阵)做零填充,以进行转置卷积
        if hd - hk + 1 != hx:
            pad = (hx - hd + hk - 1) // 2
            pad_delta = np.pad(
                delta, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
        else:
            pad_delta = delta

        for i in range(bsize):
            # 生成输出矩阵的小图像矩阵
            pad_delta_col = self.img2col(pad_delta[i],
                                         wk, self.stride)  # 784,72
            # 计算误差
            delta_backward[i] = (pad_delta_col @ k_180_col
                                 ).reshape(wx, hx, ck)  # 28,28,1

        # 反向传播
        # 使用惩罚函数更新卷积核

        self.k -= self.k_gradient * lr
        self.b -= self.b_gradient * lr

        return delta_backward


# pool
class Pool(object):
    def forward(self, x):
        bsize, w, h, c = x.shape
        new_w, new_h = w // 2, h // 2
        feature = np.zeros((bsize, new_w, new_h, c))

        # 记录最大池化时最大值的位置信息(mask:掩码)用于反向传播
        self.feature_mask = np.zeros((bsize, w, h, c))
        for bi in range(bsize):
            for ci in range(c):
                for i in range(new_w):
                    for j in range(new_h):
                        feature[bi, i, j, ci] = np.max(
                            x[bi, i * 2:i * 2 + 2, j * 2:j * 2 + 2, ci])
                        index = np.argmax(
                            x[bi, i * 2:i * 2 + 2, j * 2:j * 2 + 2, ci])
                        self.feature_mask[bi, i * 2 + index // 2,
                                          j * 2 + index % 2, ci] = 1
        return feature  # 3,13,13,8

    def backward(self, delta):
        # 向1,2两个轴扩展维度:3,13,13,8->3,26,26,8
        # 并与存储最大值位置的矩阵做Hadamard积
        return np.repeat(
            np.repeat(delta, 2, axis=1), 2, axis=2) * self.feature_mask


# # Relu
# class Relu(object):
    # def forward(self, x):
    #     self.x = x
    #     return np.maximum(x, 0)

    # def backward(self, delta):
    #     delta[self.x < 0] = 0
    #     return delta


# 全连接层
class Dense(object):
    def __init__(self, insize, outsize):
        scale = np.sqrt(insize / 2)
        self.W = np.random.randn(insize, outsize) / scale  # 1352,10
        self.b = np.random.randn(outsize) / scale  # 10,
        self.W_gradient = np.zeros((insize, outsize))
        self.b_gradient = np.zeros(outsize)

    def forward(self, x):
        self.xshape = x.shape  # 3,13,13,8
        self.x = x.reshape(self.xshape[0], -1)  # 3,1352
        return self.x @ self.W + self.b  # 3,10

    def backward(self, delta, lr):
        # delta: 3,10
        # 梯度计算
        bsize = self.x.shape[0]  # 3
        self.W_gradient = self.x.T @ delta / bsize  # 1352,10
        self.b_gradient = np.sum(delta, axis=0) / bsize  # 10,
        delta_backward = delta @ self.W.T  # 3,1352
        # 反向传播
        self.W -= self.W_gradient * lr
        self.b -= self.b_gradient * lr

        return delta_backward.reshape(self.xshape)


# Softmax
class Softmax(object):
    def cal_loss(self, predict, label):
        bsize, _ = predict.shape  # 3,10
        self.predict(predict)
        loss = 0
        delta = np.zeros(predict.shape)
        for i in range(bsize):
            delta[i] = self.softmax[i] - label[i]
            # 独热编码后,直接进行向量点乘即可得到损失
            loss -= np.sum(np.log(self.softmax[i]) * label[i])
        loss /= bsize
        return loss, delta

    def predict(self, predict):
        bsize, _ = predict.shape  # 3,10
        self.softmax = np.zeros(predict.shape)

        for i in range(bsize):
            predict_tmp = np.exp(predict[i] - np.max(predict[i]))
            # 计算batch中每一个元素的softmax概率并返回
            self.softmax[i] = predict_tmp / np.sum(predict_tmp)
        return self.softmax  # 3,10


def train(train_images, train_labels, ksize=3,
          bsize=3, lr=.005, epochs=3, knum=8):
    # 训练
    conv = Conv(kshape=(ksize, ksize, 1, knum))  # 26x26x8
    pool = Pool()                         # 13x13x8
    psize = (28 - ksize + 1) // 2
    dense = Dense(psize ** 2 * knum, 10)
    softmax = Softmax()

    for epoch in range(epochs):
        for i in range(0, len(train_images), bsize):
            X = train_images[i:i + bsize]
            Y = train_labels[i:i + bsize]

            # 前向传播过程
            predict = conv.forward(X)  # 3,26,26,8
            predict = pool.forward(predict)  # 3,13,13,8
            predict = dense.forward(predict)  # 3,10
            loss, delta = softmax.cal_loss(predict, Y)

            # 反向传播过程
            delta = dense.backward(delta, lr)
            delta = pool.backward(delta)
            conv.backward(delta, lr)

            print("Epoch-{}-{:05d}".format(str(epoch + 1), i + bsize),
                  ":", "loss:{:.4f}".format(loss))
        lr *= .95**(epoch + 1)
        # 存储模型训练的权重信息，方便测试模型时调用
        np.savez("params.npz", k=conv.k, b=conv.b, W=dense.W, nb=dense.b)


def eval(test_images, test_labels, ksize=3, bsize=3, knum=8):
    # 加载权重信息
    r = np.load("params.npz")

    conv = Conv(kshape=(ksize, ksize, 1, knum))  # 26x26x8
    pool = Pool()  # 13x13x8
    psize = (28 - ksize + 1) // 2
    dense = Dense(psize ** 2 * knum, 10)
    softmax = Softmax()

    conv.k = r["k"]
    conv.b = r["b"]
    dense.W = r["W"]
    dense.b = r["nb"]

    num = 0
    for i in range(len(test_images)):
        X = test_images[i]
        X = X[np.newaxis]  # 扩展数组维度(图像需要变为4轴)
        Y = test_labels[i]

        # 前向传播计算梯度
        predict = conv.forward(X)
        predict = pool.forward(predict)
        predict = dense.forward(predict)
        predict = softmax.predict(predict)

        if np.argmax(predict) == Y:
            num += 1

    print("TEST-ACC: ", num / len(test_images) * 100, "%")


if __name__ == '__main__':
    # 导入数据（image及label）
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        training, validation, test = pickle.load(f, encoding='bytes')

    # 训练数据(total:50000)
    tr = 9000
    shuffle1 = np.random.permutation(tr)
    train_images = training[0][:tr].reshape(tr, 28, 28, 1)[shuffle1]
    # 标签one-hot处理
    train_labels = onehot(training[1][:tr], tr)[shuffle1]

    # 测试数据(total:10000)
    te = 100
    shuffle2 = np.random.permutation(te)
    test_images = test[0][:te].reshape(te, 28, 28, 1)[shuffle2]
    test_labels = test[1][:te][shuffle2]

    print("训练模型")
    train(train_images, train_labels)

    print("测试模型")
    eval(test_images, test_labels)
