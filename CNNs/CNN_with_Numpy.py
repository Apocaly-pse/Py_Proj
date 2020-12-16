import gzip
import pickle
import numpy as np

"""
使用Numpy数组处理库实现卷积神经网络识别手写数字(MNIST)
使用三个类(Conv、Maxpool、Softmax)进行前向和反向传播
参考：https://zhuanlan.zhihu.com/p/102119808
"""


class Conv:
    # 实现卷积层类（卷积核大小：shape）

    def __init__(self, shape, num_kernels):
        # 设定卷积核个数
        self.num_kernels = num_kernels
        self.kernel_size = shape

        # 使用随机数函数初始化卷积核（相当于全连接的权重）
        # 除以9使得权重不会过大而导致学习效率降低
        self.kernels = np.random.randn(num_kernels, shape, shape) / shape ** 2

    def iterate_imgs(self, image):
        # 用于生成与卷积核作用的小图像矩阵
        height, width = image.shape

        for i in range(height - self.kernel_size + 1):
            for j in range(width - self.kernel_size + 1):
                iter_img = image[i:(i + self.kernel_size), j:(j + self.kernel_size)]
                # 返回含有位置信息（左上角位置）和矩阵的生成器
                yield iter_img, i, j

    def forward(self, input):
        # 卷积层的前向传播过程
        # 使用离散卷积将卷积核作用于输入图像矩阵(28*28)
        # `last_input`用于反向传播的梯度计算
        self.last_input = input
        height, width = input.shape

        output = np.zeros((height - self.kernel_size + 1, width - self.kernel_size + 1, self.num_kernels))

        # 进行离散卷积操作（使用点乘）
        for iter_img, i, j in self.iterate_imgs(input):
            # 这里在1,2方向上进行求和，由于0方向是num_kernels
            output[i, j] = np.sum(iter_img * self.kernels, axis=(1, 2))

        return output

    def backprop(self, nabla_out, lr):
        # 更新卷积核（本质是权重）
        # nabla_out: 池化层输出的损失函数梯度
        nabla_kernels = np.zeros(self.kernels.shape)
        # 遍历前面生成的小图像矩阵及索引
        for iter_img, i, j in self.iterate_imgs(self.last_input):
            # 逐个计算，一次更新一个卷积核
            for n in range(self.num_kernels):
                # 对小图像矩阵求和(遍历ij)即可得到最后的梯度值
                nabla_kernels[n] += nabla_out[i, j, n] * iter_img

        # 更新卷积核
        self.kernels -= lr * nabla_kernels
        # 卷积层是模型的第一层，不存在输入的损失
        # 所以不需要返回损失函数的梯度
        return None


class MaxPool:
    # 最大值池化层类（池化大小：2*2）

    def __init__(self, size):
        self.pool_size = size

    def iterate_imgs(self, image):
        # 与卷积层类似，首先生成用于池化操作的小图像矩阵
        height, width, _ = image.shape
        new_height = height // self.pool_size
        new_width = width // self.pool_size

        for i in range(new_height):
            for j in range(new_width):
                iter_img = image[(i * self.pool_size):(i * self.pool_size + self.pool_size), (j * self.pool_size):(j * self.pool_size + self.pool_size)]
                # 返回小图像矩阵及位置的生成器
                yield iter_img, i, j

    def forward(self, input):
        # 正向传播过程，使用最大值函数进行池化操作

        self.last_input = input
        # 此时的输入来自卷积层的输出
        height, width, num_kernels = input.shape
        # 先初始化输出矩阵
        output = np.zeros((height // self.pool_size, width // self.pool_size, num_kernels))

        # 开始计算最大值，使用`amax()`函数
        for iter_img, i, j in self.iterate_imgs(input):
            # 针对0,1方向计算最大值，并返回输出值
            output[i, j] = np.amax(iter_img, axis=(0, 1))

        return output

    def backprop(self, nabla_out):
        # nabla_input: 池化层输出的损失函数梯度
        # 由于池化层不存在权重和偏置，仅计算损失对输入的梯度
        nabla_input = np.zeros(self.last_input.shape)

        # 遍历前面生成的小图像矩阵及索引，找到最大值并赋值梯度
        for iter_img, i, j in self.iterate_imgs(self.last_input):
            height, width, num_kernels = iter_img.shape
            # 找到生成的小图像矩阵中元素的最大值
            # 参数axis设定为(0,1)使最大值个数为卷积核个数
            max_val = np.amax(iter_img, axis=(0, 1))
            for h, w, n in zip(range(height), range(width), range(num_kernels)):
                # 如果像素点为最大值，将梯度赋给它
                if iter_img[h, w, n] == max_val[n]:
                    # (h,w,n)为最大值的相对位置(13x13x8中)
                    # (i+h,j+w,n)为最大值的绝对位置(26x26x8中)
                    nabla_input[i + h, j + w, n] = nabla_out[i, j, n]
            # 返回损失函数关于输入的梯度
            return nabla_input


class Softmax:
    # 使用Softmax激活的全连接层类

    def __init__(self, input_dim, nodes):
        # 初始化权重和偏置
        # input_dim: 13x13x8=1352
        #   nodes  : 10
        self.weights = np.random.randn(input_dim, nodes) / input_dim
        self.biases = np.zeros(nodes)

    def forward(self, input):
        # 进行前向传播，用Softmax函数计算激活值
        self.last_input_shape = input.shape

        # 将输入（图像矩阵）展平为13*13*8=1352维向量
        input = input.flatten()
        self.last_input = input

        # weights形状：1352*10，10为待识别输出的节点数
        # 计算带权输出total
        total = input @ self.weights + self.biases
        self.last_total = total

        exp_total = np.exp(total)
        # 计算Softmax激活值并返回
        return exp_total / np.sum(exp_total, axis=0)

    def backprop(self, nabla_out, lr):
        # 首先对Softmax层(输出层,全连接层)进行反向传播
        # 注意这里的nabla_out是损失函数关于输出out的梯度(偏导数)
        for i, gradient in enumerate(nabla_out):
            # 判断gradient是否为0，为0则执行循环下一步
            # k != c,梯度为0，此时未正确分类
            # k == c 梯度为1，此时正确分类
            # 当k == c时尝试找到i
            if gradient == 0:
                continue

            t_exp = np.exp(self.last_total)

            # k!=c情形，先全部赋值给out_t,后再单独修改k==c情形的值
            out_t = -t_exp[i] * t_exp / (np.sum(t_exp) ** 2)
            out_t[i] = t_exp[i] * (np.sum(t_exp) - t_exp[i]) / (np.sum(t_exp) ** 2)
            # 计算输入的带权输出关于权重和偏置的偏导数
            t_w = self.last_input
            t_b = 1
            # 1352x10
            t_inputs = self.weights

            # 根据链式法则可得
            # (10, ) * (10, ) -> (10, )
            nabla_t = gradient * out_t
            # 用np.newaxis（或None）扩展数组为二维
            # 并进行转置操作后进行矩阵乘法运算
            # (1352, 1) @ (1, 10) -> (1352, 10)
            nabla_w = t_w[np.newaxis].T @ nabla_t[np.newaxis]
            # (10, ) * 1 -> (10, )
            nabla_b = nabla_t * t_b

            # (1352, 10) @ (10, ) -> (1352, )
            nabla_inputs = t_inputs @ nabla_t

            # 使用计算出来的nabla_w和nabla_b更新权重和偏置
            self.weights -= lr * nabla_w
            self.biases -= lr * nabla_b

            # 返回损失函数关于输入的梯度，用于池化层的backprop
            # 1352 -> 13x13x8
            return nabla_inputs.reshape(self.last_input_shape)


def Forward(image, label):
    # 定义函数用于模型的前向传播过程，输出交叉熵损失函数值及精度
    # 原始数据是(784, 1)格式的图片，像素值为[0, 1]，需要进行转换
    output1 = conv.forward(image - 0.5)
    output2 = pool.forward(output1)
    # 输出值output3为向量形式(10,1)的概率，最大概率即模型判定的数字
    output3 = softmax.forward(output2)

    # 计算交叉熵损失及精度
    loss = -np.log(output3[label])
    # 正确返回1，否则为0
    accuracy = 1 if np.argmax(output3) == label else 0

    return output3, loss, accuracy


def Train(image, label, lr=.05):
    # 首先进行前向传播，lr为学习率默认值
    output, loss, accuracy = Forward(image, label)
    # 计算初始梯度
    gradient = np.zeros(10)
    gradient[label] = -1 / output[label]

    # 计算损失梯度，并进行反向传播
    gradient1 = softmax.backprop(gradient, lr)
    gradient2 = pool.backprop(gradient1)
    conv.backprop(gradient2, lr)

    return loss, accuracy


# 导入数据（image及label）
with gzip.open('mnist.pkl.gz', 'rb') as f:
    training, validation, test = pickle.load(f, encoding='latin1')

# 训练数据(total:50000)
train_images = training[0][:10000]
train_labels = training[1][:10000]
# 测试数据(total:10000)
test_images = test[0][:1000]
test_labels = test[1][:1000]

# ----------模型调用-----------
# 实例化三个层
kernel_size = 4
kernel_num = 8
maxpool_size = 2
softmax_nodes = (28 - kernel_size + 1) // maxpool_size

conv = Conv(kernel_size, kernel_num)
pool = MaxPool(maxpool_size)
softmax = Softmax(pow(softmax_nodes, 2) * kernel_num, 10)


print("开始训练模型")

# 使用np.random.permutation函数打乱（shuffle）训练数据
shuffle = np.random.permutation(len(train_images))
train_images = train_images[shuffle]
train_labels = train_labels[shuffle]

# 初始化损失和正确识别个数
loss = 0
num_correct = 0
batch_size = 10

mini_batches = [train_images[k:k + batch_size] for k in range(0, len(train_images), batch_size)]


for i, (img, label) in enumerate(zip(train_images, train_labels)):
    # 每100次输出精确度和损失值
    if i % 100 == 99:
        print(
            '[Step %d] 前100次平均损失: %.3f | Accuracy: %d%%' %
            (i + 1, loss / 100, num_correct)
            )
        loss = 0
        num_correct = 0

    # 由于输入图片形状是(784,1)，需要转换为28x28
    los, accuracy = Train(img.reshape(28, 28), label)
    loss += los
    num_correct += accuracy


print("开始测试模型")

loss = 0
num_correct = 0
for img, label in zip(test_images, test_labels):
    _, los, accuracy = Forward(img.reshape(28, 28), label)
    loss += los
    num_correct += accuracy

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)
