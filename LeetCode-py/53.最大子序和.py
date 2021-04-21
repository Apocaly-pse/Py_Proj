#
# @lc app=leetcode.cn id=53 lang=python3
#
# [53] 最大子序和
#

# @lc code=start
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 直观的想法:可以实现但是运行时间过慢
        # ret = max(nums)
        # n = len(nums)
        # if n==1:
        #     ret = nums[0]
        # else:
        #     for i in range(n):
        #         for j in range(i, n):
        #             if sum(nums[i:j+1]) > ret:
        #                 ret = sum(nums[i:j+1])
        # return ret
        # 类比双指针的思想,动态计算并更新求出的和的最大值
        res, tmp_sum=nums[0], 0
        for num in nums:
            tmp_sum = max(tmp_sum+num, num)
            res = max(tmp_sum, res)
        return res
            
# @lc code=end

