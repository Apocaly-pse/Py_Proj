"""
快速排序:
采用`递归`的方法, 使用左右两个指针不断向中间遍历,
判断指针指向的元素与pivot的大小, 并进行交换,
逐渐减小问题规模, 以达到排序目的
"""


def QuickSort(a):
    # 首先指定pivot
    pivot = a[0]
    n = len(a)
    
    return a


a = [54, 26, 93, 17, 77, 31, 44, 55, 20]

print(a)
print(QuickSort(a))
