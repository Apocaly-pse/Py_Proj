#
# @lc app=leetcode.cn id=7 lang=python3
#
# [7] 整数反转
#

# @lc code=start
class Solution:
    def reverse(self, x: int) -> int:
        if x < 0:
            tmp = int('-' + str(x)[::-1][:-1])
        else:
            tmp = int(str(x)[::-1])
        return tmp if tmp >= -2 << 30 and tmp <= (2 << 30) - 1 else 0

# @lc code=end
