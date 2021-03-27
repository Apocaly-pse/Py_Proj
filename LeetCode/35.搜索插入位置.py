#
# @lc app=leetcode.cn id=35 lang=python3
#
# [35] 搜索插入位置
#

# @lc code=start
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        # 若target存在于nums
        if nums.count(target) != 0:
            return nums.index(target)
        else:
            # 若不存在target
            if nums[0] >target:
                # 头尾需要另外考虑
                return 0
            elif nums[len(nums)-1] <target:
                return len(nums)
            else:
                for i in range(len(nums)):
                    if nums[i-1]<target and nums[i]>target:
                        return i
# @lc code=end

