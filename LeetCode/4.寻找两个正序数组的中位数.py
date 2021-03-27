#
# @lc app=leetcode.cn id=4 lang=python3
#
# [4] 寻找两个正序数组的中位数
#

# @lc code=start
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        nums3 = nums1 + nums2
        nums3.sort()
        n = len(nums3)
        if n%2==0:
            # 偶数
            return (nums3[n//2-1]+nums3[n//2])/2
        else:
            return nums3[n//2]
# @lc code=end

