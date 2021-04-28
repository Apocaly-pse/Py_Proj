#
# @lc app=leetcode.cn id=168 lang=python3
#
# [168] Excel表列名称
#

# @lc code=start
class Solution:
    def convertToTitle(self, n: int) -> str:
        s = ''
        while n:
            n-=1
            s=chr(65+n%26)+s
            n//=26
        return s
# @lc code=end
