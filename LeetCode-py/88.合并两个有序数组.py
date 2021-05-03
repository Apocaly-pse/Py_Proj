#
# @lc app=leetcode.cn id=88 lang=python3
#
# [88] 合并两个有序数组
#

# @lc code=start
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        # 若需要在原值进行改变,需要在引用时候加上[:]
        nums1[:] = nums1[:m] + nums2[:n]
        nums1.sort()
# @lc code=end
