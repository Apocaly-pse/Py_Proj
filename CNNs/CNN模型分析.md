[toc]

# CNN 模型分析

## 模型结构

> 使用Python第三方库`numpy`编写基本的卷积神经网络(CNN)模型，主要结构如下：
>
> 1. 卷积层 (Conv)
> 2. 最大值池化层 (Pool)
> 3. 全连接层 (Dense)
> 4. Softmax层 (Softmax)
>
> 

## 超参数的选择

|   参数名称 (变量名)    |    参数值     |        备注        |
| :--------------------: | :-----------: | :----------------: |
|   batch size (bsize)   |      $3$      |   小批量数据大小   |
|   learning rate (lr)   |    $0.005$    |       学习率       |
|  Kernel size (ksize)   |  $3\times3$   |     卷积核大小     |
|  Kernel number (knum)  |      $8$      | 卷积核个数,通道数  |
|   Kernel bias (bias)   | $\mbox{True}$ |   添加卷积层偏置   |
|   Pool size (psize)    |  $2\times2$   |   最大值池化大小   |
|     padding (pad)      |      $0$      | 输入图像零填充大小 |
| Kernel stride (stride) |      $1$      |   卷积核移动步长   |
|  Pool stride (stride)  |      $2$      |      池化步长      |
|     epoch (epochs)     |      $3$      |       迭代期       |
|        shuffle         | $\mbox{True}$ |    打乱输入数据    |



## 前向传播参数维度变化(batch=3)

反向传播类似，方向相反。

```flow
Input=>start: 输入数据: (3,28,28,1)
conv=>operation: 卷积层(3,26,26,8)
pool=>operation: 池化层(3,13,13,8) 
flatten=>operation: Flatten处理(3,1352)
nn=>operation: 全连接层(3,10)
softmax=>operation: Softmax层(10,)
e=>end: 输出识别结果
Input()->conv(right)->pool(right)->nn(right)->softmax->e
```



## 代码改进方案

1. 增加mini_batch处理，提高运行效率；
2. Softmax层计算梯度并更新；
3. 增加修正线性单元(ReLU)，防止梯度消失&爆炸；
4. 输入数据增加零填充；
5. 使用独热向量编码，便于误差计算；
6. 循环所得结果使用生成器存储；
7. 



## 创新点

1. 将小图像矩阵以列表形式存储，与卷积核作用时可以利用矩阵乘法，提高运行效率，便于mini_batch操作；
2. 池化层中得到的最大值可以在图像上表示为1，再与扩展维度的输出矩阵进行Hadamard积，便于反向传播的计算；
3. 使用全连接层进行降维处理，提高模型精度；
4. 每一迭代期后相应减小学习率(\*=0.95)，提高模型训练效果；
5. 



## 需要注意的地方

1. 注意各层之间输入输出流的**维数**，尤其是多层之间通道数改变的情况；
2. 选取的`batch_size`必须**整除**输入数据的大小，否则会报错；
3. 要注意循环返回值(`return`)的**缩进**，否则会导致数据不完整而使模型训练失效；
4. 矩阵乘积运算时注意需要对哪个矩阵进行**转置**；
5. 需要注意图形的维数比卷积核小，运算时需要扩展维数



## 其他事项
1. 