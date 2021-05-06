"""
栈的实现
"""


class Stack(object):
    """
    1. 添加新元素item到栈顶: 压栈 push(item)
    2. 栈顶弹出元素: 出栈 pop()
    3. 返回栈顶元素: peek()
    4. 判断栈是否为空: is_empty()
    5. 返回栈的元素个数: size()
    """

    def __init__(self):
        # 初始化私有变量
        self.__list = []

    def push(self, item):
        self.__list.append(item)

    def pop(self):
        return self.__list.pop()

    def peek(self):
        if self.is_empty():
            return None
        else:
            return self.__list[-1]

    def is_empty(self):
        # return self.__list == []
        return not self.__list

    def size(self):
        return len(self.__list)


if __name__ == '__main__':
    s = Stack()
    s.push(1)
    s.push(3)
    print(s.peek())
    print(s.pop())
    print(s.pop())
    print(s.peek())
    print(s.is_empty())
    print(s.size())
