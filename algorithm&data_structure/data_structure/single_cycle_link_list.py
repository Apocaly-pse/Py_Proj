# coding:utf-8

"""
单链表类的Python实现
"""


class Node(object):
    # 结点类，用于保存上一节点的信息
    def __init__(self, elem):
        self.elem = elem
        self.next = None


class SingleCycleLinkList(object):
    # 单链表类
    def __init__(self, node=None):
        # 保存头结点的信息
        self.__head = node
        if node:
            pass

    def is_empty(self):
        """判断链表是否为空"""
        return self.__head is None

    def length(self):
        """单链表长度"""
        # cur=current:用于记录当前位置
        # 遍历各节点
        cur = self.__head
        # count 用于计数
        count = 0
        while cur is not None:
            count += 1
            # 向后移动
            cur = cur.next
        return count

    def travel(self):
        """遍历整个单链表并打印"""
        # cur=current:用于记录当前位置,遍历各节点
        cur = self.__head
        while cur is not None:
            print(cur.elem, end=" ")
            # 向后移动
            cur = cur.next
        print()

    def add(self, item):
        """单链表头部添加元素, 头插法"""
        node = Node(item)
        # 新生成的节点尾部指向原有头结点指向的信息
        node.next = self.__head
        self.__head = node

    def append(self, item):
        """链表尾部添加元素, 尾插法"""
        node = Node(item)
        # 进行判断,特殊处理（空链表cur无next属性）
        if self.is_empty():
            self.__head = node
        else:
            cur = self.__head
            while cur.next is not None:
                cur = cur.next
            cur.next = node

    def insert(self, pos, item):
        """
        链表指定位置添加元素
        :param  pos从零开始
        """
        if pos <= 0:
            self.add(item)
        elif pos > (self.length() - 1):
            self.append(item)
        else:
            pre = self.__head
            count = 0
            while count < (pos - 1):
                count += 1
                pre = pre.next
            # 循环退出后，pre指针指向pos-1位置
            node = Node(item)
            node.next = pre.next
            pre.next = node

    def remove(self, item):
        """删除结点(从头开始查找并删除，需要进行遍历)"""
        cur = self.__head
        pre = None
        while cur is not None:
            if cur.elem == item:
                # 判断是否为头结点
                if cur == self.__head:
                    self.__head = cur.next
                else:
                    pre.next = cur.next
                break
            else:
                # 向后遍历
                pre = cur
                cur = cur.next

    def search(self, item):
        """搜索结点,若存在则返回真"""
        cur = self.__head
        while cur is not None:
            if cur.elem == item:
                return True
            else:
                cur = cur.next
        return False


if __name__ == '__main__':
    scll = SingleCycleLinkList()
    # print(scll.is_empty())
    # print(scll.length())
    scll.append(11)

    # print(scll.is_empty())
    # print(scll.length())

    scll.append(22)
    scll.append(23)
    scll.append(24)
    scll.add(100)
    scll.append(26)
    scll.append(27)

    # print(scll.length())
    scll.insert(3, 200)
    scll.travel()
    scll.remove(200)
    scll.remove(11)
    scll.remove(100)
    scll.travel()
