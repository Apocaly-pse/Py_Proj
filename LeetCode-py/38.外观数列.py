#
# @lc app=leetcode.cn id=38 lang=python3
#
# [38] 外观数列
#

# @lc code=start
class Solution:
    def countAndSay(self, n: int) -> str:
        if n==1:
            return '1'
        else:
            return self.countAndSay()

# @lc code=end

