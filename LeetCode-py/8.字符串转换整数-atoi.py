#
# @lc app=leetcode.cn id=8 lang=python3
#
# [8] 字符串转换整数 (atoi)
#

# @lc code=start
class Solution:
    def myAtoi(self, s: str) -> int:
        s = s.strip()
        isNegative = False
        # for i in range(len(s)):
        if s.find('-'):
            isNegative = True
        

# @lc code=end

