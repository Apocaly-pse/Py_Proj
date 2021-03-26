#
# @lc app=leetcode.cn id=69 lang=python3
#
# [69] x 的平方根
#

# @lc code=start
class Solution:
    def mySqrt(self, x: int) -> int:
        if x==0:
            return 0
        elif x==1:
            return 1
        else:
            for i in range(x//2+1):
                if i**2<=x and (i+1)**2>x:
                    break
            return i
# @lc code=end

