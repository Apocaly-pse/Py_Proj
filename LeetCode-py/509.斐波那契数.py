#
# @lc app=leetcode.cn id=509 lang=python3
#
# [509] 斐波那契数
#

# @lc code=start
class Solution:
    def fib(self, n: int) -> int:
        # # 递归解法(内存占用相对较小,运行时间较慢)
        # if n>2:
        #     return self.fib(n-1) + self.fib(n-2)
        # elif n==0:
        #     return 0
        # else:
        #     return 1

        # # 循环解法
        # ret = [0, 1]
        # for i in range(1, n):
        #     ret.append(ret[i]+ret[i-1])
        # return ret[n]

        # # 循环解法:改进,降低内存占用
        # ret = [0, 1]
        # for i in range(n-1):
        #     ret.append(ret[0] + ret[1])
        #     ret.pop(0)
        # if n == 0:
        #     return 0
        # else:
        #     return ret[1]

        # best solve
        a, b=0, 1
        for i in range(n):
            a,b = b, a+b
        return a


# @lc code=end

