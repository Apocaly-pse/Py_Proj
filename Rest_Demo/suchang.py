import numpy as np

# np.set_printoptions(precision=4)

f_ij = np.array([[0, 3, 3, 1, 1],
                 [0, 3, 4, 3, 2],
                 [2, 4, 1, 2, 2],
                 [2, 2, 0, 2, 2],
                 [3, 1, 3, 0, 0]])
# print(f_ij.shape);exit()
p_ij = np.array([[0, 3 / 8, 3 / 8, 1 / 8, 1 / 8],
                 [0, 3 / 12, 4 / 12, 3 / 12, 2 / 12],
                 [2 / 11, 4 / 11, 1 / 11, 2 / 11, 2 / 11],
                 [2 / 8, 2 / 8, 0, 2 / 8, 2 / 8],
                 [3 / 7, 7, 3 / 7, 0, 0]
                 ])
# print(p_ij.shape);exit()
p_j = np.array([7 / 46, 13 / 46, 11 / 46, 8 / 46, 7 / 46])

# print(f_ij[4, 1] * np.abs(np.log(p_ij[4, 1] / p_j[1])));exit()

ret=0
for i in range(5):
    for j in range(5):
        if p_ij[i, j] != 0:
            print(
                round(f_ij[i, j] * np.abs(np.log(p_ij[i, j] / p_j[j])), ndigits=4))
            a=round(f_ij[i, j] *
                    np.abs(np.log(p_ij[i, j] / p_j[j])), ndigits=4)
            ret += a
print(ret*2)
