# V1: O(N^2)
# def BubbleSort(a):
#     n = len(a)
#     for j in range(n - 1):
#         # 遍历次数
#         for i in range(n - j - 1):
#             # 每次从头到尾遍历
#             if a[i] > a[i + 1]:
#                 a[i], a[i + 1] = a[i + 1], a[i]
#     return a


# V2: 使用flag进行改进 O(N)
def BubbleSort(a):
    n = len(a)
    for j in range(n - 1):
        # 遍历次数
        count = 0
        for i in range(n - j - 1):
            # 每次从头到尾遍历
            if a[i] > a[i + 1]:
                a[i], a[i + 1] = a[i + 1], a[i]
                count += 1
        if count == 0:
            return a
    return a


a = [54, 26, 93, 17, 77, 31, 44, 55, 20]

print(a)
print(BubbleSort(a))
