import numpy as np
# import matplotlib.pyplot as plt
import time
# 生成Toeplitz矩阵
from scipy.linalg import toeplitz
# 生成块对角矩阵
from scipy.linalg import block_diag
# 计算矩阵的奇异值
from numpy.linalg import svd

"""
罚函数的实现(v1.0)
采用罚函数方法修正卷积核参数
缺点与不足：
1. 不支持图像通道数g>1的情况
2. 代码的M矩阵生成部分及G的计算部分效率有待提高
3. 参数是否为全局参数需要考虑
4. 卷积核形状k是否可以任意取值
"""

# 设置输出值位数：全部输出
np.set_printoptions(threshold=np.inf)


class Penalty_func(object):
    def __init__(self, x, k, alpha=1):
        self.k = k  # 卷积核
        self.N = x.shape[0]  # 图像大小
        self.alpha = alpha
        self.eye = np.eye(x.shape[-1] * self.N ** 2)

    def generateM(self, k):
        # 用于生成矩阵M(hN^2,gN^2)
        # A矩阵中行(列)非零元个数
        num = k.shape[0] // 2 + 1  # 2

        # # 图像通道数
        # g = k.shape[2]
        # 卷积核个数（通道数）
        h = k.shape[-1]

        # 初始化Toeplitz矩阵: A0, Am1, A1
        # c,r分别代表各A矩阵的首列(c),首行(r)
        c0, r0, cm1, rm1, c1, r1 = np.zeros((6, self.N, h))

        # 生成首行和首列, (N,h)
        # [::-1]代表元素换序, 这里只需对列进行操作
        c0[:num] = k[:, 1, 0][:num][::-1]
        r0[:num] = k[:, 1, 0][-num:]
        cm1[:num] = k[:, 2, 0][:num][::-1]
        rm1[:num] = k[:, 2, 0][-num:]
        c1[:num] = k[:, 0, 0][:num][::-1]
        r1[:num] = k[:, 0, 0][-num:]

        # 生成1组（每组h个）Toeplitz矩阵: (h,N,N)
        A0 = np.array([toeplitz(c0[:, i], r0[:, i]) for i in range(h)])
        Am1 = np.array([toeplitz(cm1[:, i], rm1[:, i]) for i in range(h)])
        A1 = np.array([toeplitz(c1[:, i], r1[:, i]) for i in range(h)])

        # 生成块主对角阵M0，参数为重复元组（元组解包操作）(h,N^2,N^2)
        M0 = np.array([block_diag(*((A0[i], ) * self.N)) for i in range(h)])
        # 通过边补零生成另外两个块副对角阵:(h,N^2-N,N^2-N)
        Mm1 = np.array(
            [block_diag(*((Am1[i], ) * (self.N - 1))) for i in range(h)])
        M1 = np.array(
            [block_diag(*((A1[i], ) * (self.N - 1))) for i in range(h)])

        # 初始化待填充的零行和零列
        zero_col = np.zeros((h, self.N ** 2, self.N))
        zero_row = np.zeros((h, self.N, self.N * (self.N - 1)))

        Mm1 = np.dstack((zero_col, np.hstack((Mm1, zero_row))))
        M1 = np.dstack((np.hstack((zero_row, M1)), zero_col))

        # 相加得到矩阵M并返回，M:(hN^2,gN^2)
        M = np.vstack(
            [np.concatenate(mat) for mat in np.vsplit(M0 + Mm1 + M1, h)])
        return M

    def calc_G(self):
        # 计算梯度张量G，与卷积核维数一致

        # 卷积核形状
        ksz, ksz, g, h = self.k.shape
        # M的子块矩阵大小
        N2 = self.N ** 2
        # 得到矩阵M
        M = self.generateM(self.k)

        # 计算矩阵M奇异值的最大值与最小值
        print("min:", np.min(svd(M, compute_uv=False)),
              "max:", np.max(svd(M, compute_uv=False)))

        # 计算罚函数矩阵M'M-αI:(gN^2,gN^2)
        pmat = np.dot(M.T, M) - self.alpha * self.eye
        # 初始化张量G:(k,k,g,h)
        G = np.zeros(self.k.shape)

        # 遍历通道数g,h
        for h_ in range(h):
            for g_ in range(g):
                # 对M矩阵的每一个子块B(c)(d)进行遍历, 维数(N2,N2)
                for i in range(h_ * N2, h_ * N2 + N2):
                    for j in range(g_ * N2, g_ * N2 + N2):
                        if i % N2 == j % N2:
                            G[1, 1, g_, h_] += sum(pmat[j, :] * M[i, :])
                            + sum(pmat[:, j] * M[i, :])
                        elif i % N2 == j % N2 + 1:
                            G[0, 1, g_, h_] += sum(pmat[j, :] * M[i, :])
                            + sum(pmat[:, j] * M[i, :])
                        elif i % N2 == j % N2 - 1:
                            G[2, 1, g_, h_] += sum(pmat[j, :] * M[i, :])
                            + sum(pmat[:, j] * M[i, :])
                        elif i % N2 == j % N2 + self.N + 1:
                            G[0, 0, g_, h_] += sum(pmat[j, :] * M[i, :])
                            + sum(pmat[:, j] * M[i, :])
                        elif i % N2 == j % N2 + self.N:
                            G[1, 0, g_, h_] += sum(pmat[j, :] * M[i, :])
                            + sum(pmat[:, j] * M[i, :])
                        elif i % N2 == j % N2 + self.N - 1:
                            G[2, 0, g_, h_] += sum(pmat[j, :] * M[i, :])
                            + sum(pmat[:, j] * M[i, :])
                        elif i % N2 == j % N2 - self.N + 1:
                            G[0, 2, g_, h_] += sum(pmat[j, :] * M[i, :])
                            + sum(pmat[:, j] * M[i, :])
                        elif i % N2 == j % N2 - self.N:
                            G[1, 2, g_, h_] += sum(pmat[j, :] * M[i, :])
                            + sum(pmat[:, j] * M[i, :])
                        elif i % N2 == j % N2 - self.N - 1:
                            G[2, 2, g_, h_] += sum(pmat[j, :] * M[i, :])
                            + sum(pmat[:, j] * M[i, :])

        # print("总用时：", time.time() - start);exit()

        return 2 * G

    def update_kernel(self, lambd):
        # 更新卷积核参数：K -= λG
        G = self.calc_G()
        self.k -= lambd * G


if __name__ == '__main__':
    start = time.time()
    # 初始化随机数种子
    np.random.seed(1)
    x = np.random.randn(20, 20, 1)
    k = np.random.randn(3, 3, 1, 3)

    # 实例化罚函数类
    p_func = Penalty_func(x, k, alpha=1)
    # 迭代更新卷积核
    for i in range(300):
        if i < 10:
            lambd = 1e-5
        elif i < 20:
            lambd = 1e-4
        else:
            lambd = 1e-3
        p_func.update_kernel(lambd=lambd)

    print("总用时：", time.time() - start)
