#
# @lc app=leetcode.cn id=1006 lang=python3
#
# [1006] 笨阶乘
#

# @lc code=start
class Solution:
    def clumsy(self, N: int) -> int:
        ret=0
        if N>4:
            ret = N*(N-1)//(N-2)+(N-3)
            mod = N%4
            for i in range(N-4, mod, -4):
                ret -= i*(i-1)//(i-2)
                ret+=(i-3)
            if mod==3:
                ret-=6
            elif mod==2:
                ret-=2
            elif mod==1:
                ret-=1
        elif N==4:
            ret=7
        elif N==3:
            ret=6
        elif N==2:
            ret=2
        else:
            ret=1
        return ret
# @lc code=end

