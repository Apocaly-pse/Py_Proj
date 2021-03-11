# coding:utf-8

"""
双向链表类的Python实现
"""

# # 可以采用继承的方法进行复用（前四个方法）
# from single_link_list import SingleLinkList


class Node(object):
    # 结点类，用于保存上一节点的信息
    def __init__(self, elem):
        self.elem = elem
        self.next = None
        self.prev = None


class DoubleLinkList(object):
    # 双向链表类
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
        # 顺序不唯一
        self.__head = node
        node.next.prev = node

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
            node.prev = cur

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
            cur = self.__head
            count = 0
            while count < pos:
                count += 1
                cur = cur.next
            # 循环退出后，cur指针指向pos位置
            node = Node(item)
            node.next = cur
            node.prev = cur.prev
            cur.prev = node
            node.prev.next = node

    def remove(self, item):
        """删除结点(从头开始查找并删除，需要进行遍历)"""
        cur = self.__head
        while cur is not None:
            if cur.elem == item:
                # 判断是否为头结点
                if cur == self.__head:
                    self.__head = cur.next
                    if cur.next:
                        # 判断链表是否只有一个节点
                        cur.next.prev = None
                else:
                    cur.prev.next = cur.next
                    if cur.next:
                        cur.next.prev = cur.prev
                break
            else:
                # 向后遍历
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
    dll = DoubleLinkList()
    # print(dll.is_empty())
    # print(dll.length())
    dll.append(11)

    # print(dll.is_empty())
    # print(dll.length())

    dll.append(22)
    dll.append(23)
    dll.append(24)
    dll.add(100)
    dll.append(26)
    dll.append(27)

    # print(dll.length())
    dll.insert(3, 200)
    dll.travel()
    dll.remove(200)
    dll.remove(11)
    dll.remove(100)
    dll.travel()
