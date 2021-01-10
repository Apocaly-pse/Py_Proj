"""
装饰器实现
"""


def w1(func):
    def inner():
        print("checking...")
        func()
    return inner


@w1
def f1():
    print("f1")


@w1
def f2():
    print("f2")


# f2 = w1(f2)
f1()
f2()

# w1(f1)()
