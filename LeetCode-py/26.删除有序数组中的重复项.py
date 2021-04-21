#
# @lc app=leetcode.cn id=26 lang=python3
#
# [26] 删除有序数组中的重复项
#

# @lc code=start
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        tmp = list(set(nums))
        tmp.sort()
        nums[:] = tmp
        # for i in range(len(nums)-len(tmp)):
        #     nums.pop()
        nums = nums[:len(tmp)]
        return len(nums)
# @lc code=end

