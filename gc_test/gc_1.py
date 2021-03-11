import gc


class ClassA(object):
    def __init__(self):
        print('object born, id:%s' % str(hex(id(self))))


def f2():
    while 1:
        c1 = ClassA()
        c2 = ClassA()
        c1.t = c2
        c2.t = c1
        del c1
        del c2


# 如果关闭gc垃圾回收机制,则循环调用会出现内存过大导致假死现象
gc.disable()
f2()
