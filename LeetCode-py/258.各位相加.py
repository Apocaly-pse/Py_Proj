#
# @lc app=leetcode.cn id=258 lang=python3
#
# [258] 各位相加
#

# @lc code=start
class Solution:
    def addDigits(self, num: int) -> int:
        # if num==0:
        #     return 0
        # else:
        #     return num%9 if num%9!=0 else 9
        return (num-1)%9+1 if num!=0 else 0
# @lc code=end

