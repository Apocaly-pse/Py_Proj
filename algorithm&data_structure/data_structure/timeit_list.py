# coding: utf-8
# ! /opt/homebrew/bin/python3.9

# `timeit`计时器模块测算代码段的运行时间
from timeit import Timer

# li1 = [1, 2]

# li2 = [23, 5]

# li = li1 + li2

# li = [i for i in range(10000)]

# li = list(range(10000))

def test1():
    li = []
    for i in range(10000):
        li.append(i)


def test2():
    li = []
    for i in range(10000):
        # li += [i]
        li = li + [i]

# python 针对+=运算符进行了优化，所以时间数量级与extend类似，但是直接采用= +的话速度会变得很慢

"""
append: 0.37060525
+: 89.167177542
generator: 0.17624683300000754
list: 0.08445720800000345
list_extend: 0.5095627500000006

***Repl Closed***

"""

def test3():
    li = [i for i in range(10000)]


def test4():
    li = list(range(10000))


def test5():
    li = []
    for i in range(10000):
        li.extend([i])


# 构建待测算的Timer类
timer1 = Timer("test1()", "from __main__ import test1")
print("append:", timer1.timeit(1000))

timer2 = Timer("test2()", "from __main__ import test2")
print("+:", timer2.timeit(1000))

timer3 = Timer("test3()", "from __main__ import test3")
print("generator:", timer3.timeit(1000))

timer4 = Timer("test4()", "from __main__ import test4")
print("list:", timer4.timeit(1000))

timer5 = Timer("test5()", "from __main__ import test5")
print("list_extend:", timer5.timeit(1000))

"""
power by m1 arm python3.9

append: 0.360514292
+: 0.383263791
generator: 0.17635270800000002
list: 0.0820765
list_extend: 0.4995229579999998

***Repl Closed***

"""


# # 使用append进行尾部追加
# def test6():
#     li = []
#     for i in range(10000):
#         li.append(i)


# # 使用insert进行头部添加
# def test7():
#     li = []
#     for i in range(10000):
#         li.insert(0, i)


# timer6 = Timer("test6()", "from __main__ import test6")
# print("append:", timer6.timeit(1000))

# timer7 = Timer("test7()", "from __main__ import test7")
# print("insert:", timer7.timeit(1000))

# """
# power by m1 arm python3.9

# append: 0.361828417
# insert: 27.604294

# ***Repl Closed***

# """