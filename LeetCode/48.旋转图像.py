#
# @lc app=leetcode.cn id=48 lang=python3
#
# [48] 旋转图像
#

# @lc code=start
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # n = len(matrix)
        # # 遍历矩阵维数
        # for i in range(n//2):
        #     for j in range(n//2):
        #         if i==0 and j==0:
        #             # 旋转角块: 仅执行一次交换, n为奇偶情况相同
        #             matrix[i][j], matrix[i][j+n-1], matrix[i+n-1][j], matrix[i+n-1][j+n-1] = matrix[i+n-1][j], matrix[i][j], matrix[i+n-1][j+n-1], matrix[i][j+n-1]
        #         if n%2 != 0:
        #             # 旋转边块: n为奇数
        #             matrix[i+n-2][j+n-1], matrix[i+n-1][j+n-2], matrix[i+n-2][j], matrix[i][j+n-2] = matrix[i][j+n-2], matrix[i+n-2][j+n-1], matrix[i+n-1][j+n-2], matrix[i+n-2][j]
        #         else:
        #             # n为偶数
        #             matrix[i+n-2][j+n-1], matrix[i+n-1][j+n-2], matrix[i+n-2][j], matrix[i][j+n-2] = matrix[i][j+n-2], matrix[i+n-2][j+n-1], matrix[i+n-1][j+n-2], matrix[i+n-2][j]
        matrix[:] = zip(*matrix[::-1])
# @lc code=end

