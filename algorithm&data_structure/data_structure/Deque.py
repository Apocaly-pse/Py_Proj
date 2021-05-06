"""
双端队列实现
Double-Ended-Queue
"""


class Deque(object):
    """docstring for Deque"""

    def __init__(self):
        self.__list = []

    def add_front(self, item):
        self.__list.insert(0, item)

    def add_rear(self, item):
        self.__list.append(item)

    def pop_front(self):
        return self.__list.pop(0)

    def pop_rear(self):
        return self.__list.pop()

    def is_empty(self):
        return not self.__list

    def size(self):
        return len(self.__list)


if __name__ == '__main__':
    q = Deque()
    q.add_front(1)
    q.add_front(2)
    q.add_front(3)
    print(q.size())
    print(q.pop_front())
    print(q.pop_front())
    print(q.is_empty())
    print(q.pop_front())
    print(q.is_empty())
