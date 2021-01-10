"""
多个装饰器的实现
"""


def w1(func):
    def inner():
        print("checking...")
        func()
    return inner


def w2(func):
    def inner():
        print("checking...222")
        func()
    return inner


# @w1
# def f1():
#     print("f1")


# @w1
# @w2
def f2():
    print("f2")


# f2 = w1(f2)
# f1()
f2 = w2(f2)
f2 = w1(f2)
# 装饰器从下往上装
f2()


# w1(f1)()
