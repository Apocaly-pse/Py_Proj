import gzip, pickle
import numpy as np

"""
使用数组处理库`numpy`编写的CNN程序
正向与反向传播
"""


class Conv3x3:
    # A convolution layer using 3x3 filters.

    def __init__(self, num_filters):
        self.num_filters = num_filters

        # filters is a 3d array with dimentions (num_filters, 3, 3)
        # We divide by 9 to reduce the variance of our initial values
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        # image: matrix of image
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        # 输入大小：28x28
        self.last_input = input

        # input_im: matrix of image
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output

    def backprop(self, d_L_d_out, learn_rate):
        # 更新卷积核的权重
        # d_L_d_out: 池化层输出的损失函数梯度
        # learn_rate: 学习速率
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            # 逐层计算，一次更新一层
            for f in range(self.num_filters):
                # d_L_d_filters[f]: 3x3 矩阵（共f个）
                # d_L_d_out[i, j, f]: [i,j,f]处的梯度值
                # im_region: 3x3 小图像矩阵
                # （代表ij位置的图像，由输入层迭代生成）
                # 直接对图像（im_region）进行求和即可得到最后的梯度值
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # 更新卷积核（filters），本质是更新卷积层的权重（类比MLP）
        self.filters -= learn_rate * d_L_d_filters

        # 卷积层是模型的第一层，而第一层不存在输入的损失
        # 所以不需要返回损失函数的梯度
        return None


class MaxPool2:
    # 池大小为2的最大值池化
    def iterate_regions(self, image):
        # image：26x26x8
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                # im_region: 13x13x8
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        '''
        Performs a forward pass of the maxpool layer using the given input.
        Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
        '''
        # input：26x26x8
        self.last_input = input

        # 输入：来自卷积层输出的三维数组
        h, w, num_filters = input.shape
        # 输出：13x13x8，寻找最大值并赋值给output
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backprop(self, d_L_d_out):
        # d_L_d_out: 池化层输出的损失函数梯度
        # d_L_d_input: 池化层不存在权重与偏置，仅计算L对输入的梯度
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            height, weight, f_num = im_region.shape
            # 找到im_region中的最大值
            # amax为8元列表(由于axis参数取值为0,1，最后剩下的最大值个数是8)
            amax = np.amax(im_region, axis=(0, 1))
            for h in range(height):
                for w in range(weight):
                    for f in range(f_num):
                        # 如果该像素点为最大值，将梯度赋值给它
                        if im_region[h, w, f] == amax[f]:
                            # (h,w,f)为最大值的相对位置（13x13x8中）
                            # (i+h,j+w,f)为最大值所在的绝对位置（26x26x8中）
                            d_L_d_input[i + h, j + w, f] = d_L_d_out[i, j, f]
        # 返回损失函数关于输入的梯度
        return d_L_d_input


class Softmax:
    # A standard fully-connected layer with softmax activation.

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        # input_len: length of input nodes
        # nodes: lenght of ouput nodes

        self.weights = np.random.randn(input_len, nodes) / input_len
        # 这里的偏置设置为零，反向传播之后进行更新
        self.biases = np.zeros(nodes)

    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''
        # 3d，13x13x8
        self.last_input_shape = input.shape

        # 3d to 1d （13x13x8=1352）
        input = input.flatten()

        # 1d vector after flatting
        self.last_input = input

        input_len, nodes = self.weights.shape

        totals = input @ self.weights + self.biases

        # output before softmax
        # 1d vector
        self.last_totals = totals

        exp_total = np.exp(totals)
        return exp_total / np.sum(exp_total, axis=0)

    def backprop(self, d_L_d_out, learn_rate):
        # only 1 element of d_L_d_out is nonzero
        for i, gradient in enumerate(d_L_d_out):
            # k != c, gradient = 0
            # k == c, gradient = 1
            # try to find i when k == c
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradients of out[i] against totals
            # all gradients are given value with k != c
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            # change the value of k == c
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of out[i] against totals
            # gradients to every weight in every node
            # this is not the final results
            d_t_d_w = self.last_input  # vector
            d_t_d_b = 1
            # 1000 x 10
            d_t_d_inputs = self.weights

            # Gradients of loss against totals
            # d_L_d_t, d_out_d_t, vector, 10 elements
            d_L_d_t = gradient * d_out_d_t

            # (1352, 1) @ (1, 10) -> (1352, 10)
            # 使用np.newaxis（None）扩展数组维数
            # 输入向量（一维）由于被flatten()函数展平，需要变成二维数组
            # 并进行转置操作后进行矩阵乘法运算
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            # (1352, 10) @ (10, 1)
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # 使用计算出来的d_L_d_w和d_L_d_b更新权重与偏置
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b

            # 1352->13x13x8，1D -> 3D，用于pooling层的backprop
            return d_L_d_inputs.reshape(self.last_input_shape)


# 数据加载部分
f = gzip.open('mnist.pkl.gz', 'rb')
training, validation, test = pickle.load(f, encoding='latin1')
f.close()

# 训练数据(total:60000)
train_images = training[0][:1000]
train_labels = training[1][:1000]

# 测试数据(total:10000)
test_images = test[0][:1000]
test_labels = test[1][:1000]

conv = Conv3x3(8)                    # 28x28x1 -> 26x26x8
pool = MaxPool2()                    # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10)    # 13x13x8 -> 10


def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.
    '''
    # 传入数据
    # 原始数据是(784, 1)格式的图片，像素值在[0, 1]，需要进行转换
    out1 = conv.forward(image.reshape(28, 28) - 0.5)
    out2 = pool.forward(out1)
    out3 = softmax.forward(out2)

    # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
    loss = -np.log(out3[label])
    acc = 1 if np.argmax(out3) == label else 0

    # out: vertor of probability
    # loss: num
    # acc: 1 or 0
    return out3, loss, acc


def train(im, label, lr=.005):
    # 前向传播
    out, loss, acc = forward(im, label)

    # 计算初始梯度
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # 损失梯度的反向传播
    gradient1 = softmax.backprop(gradient, lr)
    gradient2 = pool.backprop(gradient1)
    gradient3 = conv.backprop(gradient2, lr)

    return loss, acc


print('MNIST CNN initialized!')


# 开始训练CNN，epoch: 迭代期
for epoch in range(5):
    print('--- Epoch %d ---' % (epoch + 1))

    # 打乱（shuffle）训练数据
    # 使用np.random.permutation()函数进行随机排列处理
    shuffle = np.random.permutation(len(train_images))
    train_images = train_images[shuffle]
    train_labels = train_labels[shuffle]

    # Train
    loss = 0
    num_correct = 0
    # i: index
    # im: image
    # label: label
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i % 100 == 99:
            print(
                '[Step %d] Past 100 steps: \
                Average Loss %.3f | Accuracy: %d%%' %
                (i + 1, loss / 100, num_correct))
            loss = 0
            num_correct = 0

        loss1, acc = train(im, label)
        loss += loss1
        num_correct += acc


# 测试模型
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    _, l, acc = forward(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)
