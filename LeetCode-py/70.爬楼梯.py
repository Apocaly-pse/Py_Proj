#
# @lc app=leetcode.cn id=70 lang=python3
#
# [70] 爬楼梯
#

# @lc code=start
class Solution:
    def climbStairs(self, n: int) -> int:
        # 先计算递推式,再进行求解(等同于fib)
        ret = [0, 1]
        for i in range(1, n+1):
            ret.append(ret[i]+ret[i-1])
        return ret[n+1]
# @lc code=end

