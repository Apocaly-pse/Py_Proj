#
# @lc app=leetcode.cn id=11 lang=python3
#
# [11] 盛最多水的容器
#

# @lc code=start
class Solution:
    def maxArea(self, height: List[int]) -> int:
        # 解法1:暴力双循环,没有问题,但是时间超过定额
        # n = len(height)
        # ret = min(height[0], height[1])
        # for i in range(n):
        #     for j in range(i+1, n):
        #         if (j-i)*min(height[i], height[j]) > ret:
        #             ret = (j-i)*min(height[i], height[j])
        # return ret

        # 解法2: 双指针遍历!好办法
        res, l, r = 0, 0, len(height)-1
        while l < r:
            if height[l] < height[r]:
                # 如果左低右高,左指针右移一位(用于寻找最大的左边框)
                res, l = max(res, height[l]*(r-l)), l+1
            else:
                # 右低左高,右指针左移一位,(找最大的右边框)
                res, r = max(res, height[r]*(r-l)), r-1
        return res

# @lc code=end

