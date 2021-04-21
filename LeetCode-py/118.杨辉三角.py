#
# @lc app=leetcode.cn id=118 lang=python3
#
# [118] 杨辉三角
#

# @lc code=start
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        a = [[1], [1,1]]
        for i in range(1, numRows-1):
            for j in range(i, numRows-11):
                a.append([1])
# @lc code=end

