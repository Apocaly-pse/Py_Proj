#
# @lc app=leetcode.cn id=2 lang=python3
#
# [2] 两数相加
#

# @lc code=start
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1.val == 0 and l2.val == 0 and l1.next is None and l2.next is None:
            return l1
        else:
            ret = ListNode()
            while l1.next is not None and l2.next is not None:
                if l1.val + l2.val < 10:
                    ret.val = l1.val + l2.val
                else:
                    ret.val = 0
                # 向后遍历
                l1.val = l1.next
                l1.next = l1.val.next

        return ret

            

# @lc code=end

