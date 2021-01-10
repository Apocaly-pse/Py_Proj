"""
闭包的实现
"""


def w1(func):
    def inner():
        print("checking...")
        func()
    return inner


def f1():
    print("f1")


def f2():
    print("f2")


innf = w1(f2)
innf()

w1(f1)()
