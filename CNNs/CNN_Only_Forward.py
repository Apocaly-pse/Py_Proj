import numpy as np
import mnist

"""
仅正向传播的代码实现
结果（正确率）近似于正确的概率（10%）
"""

class Conv3x3:
    # 卷积层的类实现
    # 参数初始化
    def __init__(self, num_filters):
        # 设定卷积核（滤波器）的个数
        self.num_filters = num_filters
        # 通过随机值给定卷积核的参数
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        '''
        Generates all possible 3x3 image regions using valid padding.
        - image is a 2d numpy array
        '''
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j
        # 将 im_region, i, j 以 tuple 形式存储到迭代器中
        # 以便后面遍历使用

    def forward(self, input):
        '''
        Performs a forward pass of the conv layer using the given input.
        Returns a 3d numpy array with dimensions (h, w, num_filters).
        - input is a 2d numpy array
        '''
        # input 为 image，即输入数据
        # output 为输出框架，默认都为 0，都为 1 也可以，反正后面会覆盖
        # input: 28x28
        # output: 26x26x8
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            # 使用离散卷积运算，点乘（逐元素相乘）再求和
            # ouput[i, j] 为向量，共 8 层
            # 注意这里的sum函数的参数，需要对1，2轴进行求和（0轴是层数：8）
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        # 最后将输出数据返回，便于下一层的输入使用
        return output


class MaxPool2:
    # 使用大小为2的最大值池化。

    def iterate_regions(self, image):
        '''
        Generates non-overlapping 2x2 image regions to pool over.
        - image is a 2d numpy array
        '''
        # image: 26x26x8
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def forward(self, input):
        '''
        Performs a forward pass of the maxpool layer using the given input.
        Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
        - input is a 3d numpy array with dimensions (h, w, num_filters)
        '''
        # input: (卷积层的输出)池化层的输入
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0,1))
        return output


class Softmax:
    # A standard fully-connected layer with softmax activation.

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        # input_len: 输入层的节点个数（即池化层输出拉平之后的节点总数）
        # nodes: 输出层的节点个数，本例中为 10（分类十个数字）
        # 构建权重矩阵，初始化随机数，不能太大
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''
        # 3d to 1d，展平数组，用来构建全连接网络
        input = input.flatten()

        input_len, nodes = self.weights.shape

        # input: 13x13x8 = 1352
        # self.weights: (1352, 10)
        # 以上叉乘之后为 向量，1352个节点与对应的权重相乘再加上bias得到输出的节点
        # totals: 向量, 10
        totals = np.dot(input, self.weights) + self.biases
        # exp: 向量, 10
        exp = np.exp(totals)
        # 代入Softmax公式
        return exp / np.sum(exp, axis=0)



# 为节省时间，这里仅使用测试样例的前1000个图片（总共有10000个图片）
test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

# 数据从输入层经过卷积层后的大小变化：28x28x1 -> 26x26x8
conv = Conv3x3(8)
# 卷积层输出数据经过池化层后的大小变化：26x26x8 -> 13x13x8
pool = MaxPool2()
softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10


def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    '''

   # out1 为卷积层的输出, 26x26x8
   # 将图片数据中的像素值从[0,155]转化为[-.5,.5]，便于模型训练
    out1 = conv.forward((image / 255) - 0.5)
    # out2 为池化层的输出, 13x13x8
    out2 = pool.forward(out1)
    # out3 为 softmax 的输出, 10
    out3 = softmax.forward(out2)

    # 计算交叉熵损失与精度
    # 损失函数的计算只与 label 数有关，相当于索引（对数似然估计）
    loss = -np.log(out3[label])
    # 如果 softmax 输出的最大值就是 label 的值，表示正确，否则错误
    acc = 1 if np.argmax(out3) == label else 0
    # 前向传播返回
    return out3, loss, acc


print('MNIST CNN initialized!')

# 初始化平均损失和图片正确识别数
loss = 0
num_correct = 0
# enumerate 函数用于添加索引值
for i, (im, label) in enumerate(zip(test_images, test_labels)):
    # 进行前向传播
    out, loss1, acc = forward(im, label)
    loss += loss1
    num_correct += acc
    # print(out)

    # 每一百步打印一次结果
    if i % 100 == 99:
        print('[Step %d] Past %d steps: Average Loss %.3f | Accuracy: %d%%'
            %(i + 1, 100, loss / 100, num_correct))
        loss = 0
        num_correct = 0