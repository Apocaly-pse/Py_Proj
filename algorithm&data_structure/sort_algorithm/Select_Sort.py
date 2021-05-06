def SelectSort(a):
    # 先寻找最小的数, 然后交换最小值和首值, 以此类推
    n = len(a)
    for j in range(n - 1):
        # 遍历进行选择并交换
        k = j  # k是最小值元素的下标
        for i in range(j + 1, n):
            # 遍历寻找最小值对应的下标
            if a[k] > a[i]:
                k = i
        a[j], a[k] = a[k], a[j]
    return a


a = [54, 26, 93, 17, 77, 31, 44, 55, 20]

print(a)
print(SelectSort(a))
