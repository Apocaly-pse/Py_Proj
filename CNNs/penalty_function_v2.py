import numpy as np
# import matplotlib.pyplot as plt
import time
# 生成Toeplitz矩阵
from scipy.linalg import toeplitz
# 生成块对角矩阵
from scipy.linalg import block_diag
# 计算矩阵奇异值
from numpy.linalg import svd

"""
罚函数的实现(v2.0)
采用罚函数方法修正卷积核参数
"""

# 设置输出值位数：全部输出
np.set_printoptions(threshold=np.inf)


class Penalty_func(object):
    def __init__(self, x, k, alpha=1):
        self.k = k  # 卷积核
        self.N = x.shape[0]  # 图像大小
        g = x.shape[-1]  # 图像通道数
        self.alpha = alpha  # 参数α
        self.eye = np.eye(g * self.N ** 2)  # 单位矩阵(gN2,gN2)

    def generateM(self, k):
        # 用于生成矩阵M(hN^2,gN^2)
        # A矩阵中行(列)非零元个数
        num = k.shape[0] // 2 + 1  # 2

        # 图像通道数
        g = k.shape[2]
        # 卷积核个数（通道数）
        h = k.shape[-1]
        # M矩阵的子块矩阵大小
        N2 = self.N ** 2

        # 初始化Toeplitz矩阵(A0,Am1,A1)及首行首列信息
        # c,r分别代表各A矩阵的首列(c),首行(r)
        c0, r0, cm1, rm1, c1, r1 = np.zeros((6, self.N, g, h))
        M0, Mm1, M1 = [], [], []

        # [::-1]代表元素换序, 这里只需对列进行操作
        for i in range(g):
            # 生成首行和首列, (N,g,h)
            c0[:num, i] = k[:, 1, i][:num][::-1]
            r0[:num, i] = k[:, 1, i][-num:]
            cm1[:num, i] = k[:, 2, i][:num][::-1]
            rm1[:num, i] = k[:, 2, i][-num:]
            c1[:num, i] = k[:, 0, i][:num][::-1]
            r1[:num, i] = k[:, 0, i][-num:]
            for j in range(h):
                # 生成Toeplitz矩阵A0,Am1,A1:(N,N)
                A0 = toeplitz(c0[:, i, j], r0[:, i, j])
                Am1 = toeplitz(cm1[:, i, j], rm1[:, i, j])
                A1 = toeplitz(c1[:, i, j], r1[:, i, j])

                # 生成块主对角阵M0(g*h,N2,N2),参数为重复元组(元组解包)
                M0.append(block_diag(*((A0, ) * self.N)))
                # 通过`边补零`生成另外两个块副对角阵:(g*h,N*(N-1),N*(N-1))
                Mm1.append(block_diag(*((Am1, ) * (self.N - 1))))
                M1.append(block_diag(*((A1, ) * (self.N - 1))))

        M0, Mm1, M1 = np.array(M0), np.array(Mm1), np.array(M1)
        # 初始化待填充的零行和零列
        zero_col = np.zeros((g * h, N2, self.N))
        zero_row = np.zeros((g * h, self.N, N2 - self.N))

        # 对Mm1和M1进行补零操作,以匹配M0维数
        Mm1 = np.dstack((zero_col, np.hstack((Mm1, zero_row))))
        M1 = np.dstack((np.hstack((zero_row, M1)), zero_col))

        # 得到矩阵M并返回:(hN^2,gN^2)
        tmp_M = []
        split_M = np.vsplit((M0 + Mm1 + M1), g * h)  # (gh,N2,N2)

        # 将各子块合成为矩阵M
        for h_ in range(h):
            tmp_M.append(np.concatenate(np.block(split_M[h_::h])))
        M = np.vstack(tmp_M)

        return M

    def calc_G(self):
        # 计算梯度张量G，与卷积核维数一致

        # 卷积核形状
        ksz, ksz, g, h = self.k.shape
        # M的子块矩阵大小
        N2 = self.N ** 2
        # 计算矩阵M
        M = self.generateM(self.k)

        # 计算矩阵M奇异值的最值
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
                            G[1, 1, g_, h_] += sum(
                                (pmat[j, :] + pmat[:, j]) * M[i, :])
                        elif i % N2 == j % N2 + self.N:
                            G[1, 0, g_, h_] += sum(
                                (pmat[j, :] + pmat[:, j]) * M[i, :])
                        elif i % N2 == j % N2 - self.N:
                            G[1, 2, g_, h_] += sum(
                                (pmat[j, :] + pmat[:, j]) * M[i, :])
                        elif i % N2 == j % N2 + self.N + 1:
                            G[0, 0, g_, h_] += sum(
                                (pmat[j, :] + pmat[:, j]) * M[i, :])
                        elif i % N2 == j % N2 + 1:
                            G[0, 1, g_, h_] += sum(
                                (pmat[j, :] + pmat[:, j]) * M[i, :])
                        elif i % N2 == j % N2 - self.N + 1:
                            G[0, 2, g_, h_] += sum(
                                (pmat[j, :] + pmat[:, j]) * M[i, :])
                        elif i % N2 == j % N2 + self.N - 1:
                            G[2, 0, g_, h_] += sum(
                                (pmat[j, :] + pmat[:, j]) * M[i, :])
                        elif i % N2 == j % N2 - 1:
                            G[2, 1, g_, h_] += sum(
                                (pmat[j, :] + pmat[:, j]) * M[i, :])
                        elif i % N2 == j % N2 - self.N - 1:
                            G[2, 2, g_, h_] += sum(
                                (pmat[j, :] + pmat[:, j]) * M[i, :])

        return 2 * G

    def update_kernel(self, lambd):
        # 更新卷积核参数：K -= λG
        G = self.calc_G()
        self.k -= lambd * G


if __name__ == '__main__':
    start = time.time()
    # 初始化随机数种子
    np.random.seed(1)
    x = np.random.randn(10, 10, 6)
    k = np.random.randn(3, 3, 6, 3)

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
