#
# @lc app=leetcode.cn id=172 lang=python3
#
# [172] 阶乘后的零
#

# @lc code=start
class Solution:
    def trailingZeroes(self, n: int) -> int:
        ret = 0
        for i in range(1, n+1):
            if str(i).endswith('0'):
                ret+=1
            elif str(i).endswith('2'):
                ret+=.5
            elif i%5==0:
                ret+=.5
        return int(ret)

# @lc code=end

