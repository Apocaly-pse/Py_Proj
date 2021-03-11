# coding:utf-8

"""
单链表类的Python实现
"""


class Node(object):
    # 结点类，用于保存上一节点的信息
    def __init__(self, elem):
        self.elem = elem
        self.next = None


class SingleLinkList(object):
    # 单链表类
    def __init__(self, node=None):
        # 保存头结点的信息
        self.__head = node

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
    sll = SingleLinkList()
    # print(sll.is_empty())
    # print(sll.length())
    sll.append(11)

    # print(sll.is_empty())
    # print(sll.length())

    sll.append(22)
    sll.append(23)
    sll.append(24)
    sll.add(100)
    sll.append(26)
    sll.append(27)

    # print(sll.length())
    sll.insert(3, 200)
    sll.travel()
    sll.remove(200)
    sll.remove(11)
    sll.remove(100)
    sll.travel()
