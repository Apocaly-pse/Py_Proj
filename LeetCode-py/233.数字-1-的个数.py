#
# @lc app=leetcode.cn id=233 lang=python3
#
# [233] 数字 1 的个数
#

# @lc code=start
class Solution:
    def countDigitOne(self, n: int) -> int:
        s = str([i for i in range(1, n+1)])
        return s.count('1')
# @lc code=end

