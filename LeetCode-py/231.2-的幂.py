#
# @lc app=leetcode.cn id=231 lang=python3
#
# [231] 2的幂
#

# @lc code=start
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        # python内置的二进制转换函数,这里应用到了二进制数的性质
        return bin(n).count('1') == 1 if n>0 else False
            
        
# @lc code=end

