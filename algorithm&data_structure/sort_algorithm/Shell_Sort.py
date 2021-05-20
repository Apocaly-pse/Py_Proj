"""
希尔排序
(插入排序的改进: 分gap进行插入排序)
"""


def ShellSort(a):
    # 首先进行插排
    n = len(a)
    gap = n // 2
    # 最外层循环控制gap的size
    while gap > 0:
        for j in range(gap, n):
            # 中间的循环仍为插入排序
            for i in range(0, j, gap):
                if a[i] > a[j]:
                    a[i], a[j] = a[j], a[i]
        gap //= 2
    return a


a = [54, 26, 93, 17, 77, 31, 44, 55, 20]

print(a)
print(ShellSort(a))
