# coding:utf-8
"""
本程序用于生成排列数和组合数
"""


def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def Ank(n, k):
    return factorial(n) // factorial(n - k)


def Cnk(n, k):
    return Ank(n, k) // factorial(k)


# print(Ank(3,3)==factorial(3))
