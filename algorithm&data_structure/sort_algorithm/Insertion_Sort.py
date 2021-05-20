def InsertionSort(a):
    # 先认为第一个元素构成一个有序的序列, 然后遍历插入进行排序
    n = len(a)
    # 外层循环: 对右边无序状态的每一个元素进行遍历
    for j in range(1, n):
        for i in range(j):
            # 内层循环: 操作左边有序状态的元素进行比较和插入
            if a[i] > a[j]:
                a[i], a[j] = a[j], a[i]
    return a


a = [54, 26, 93, 17, 77, 31, 44, 55, 20]

print(a)
print(InsertionSort(a))
