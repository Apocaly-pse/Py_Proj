#
# @lc app=leetcode.cn id=73 lang=python3
#
# [73] 矩阵置零
#

# @lc code=start
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        # 依矩阵的行和列进行遍历
        # 这里需要注意:改变只能发生一次,使用原地算法
        # 如果完成了之前的置零操作可能会影响后面的0的识别
        m = len(matrix)
        n = len(matrix[0])
        index= []
        # 首先遍历得到0的索引并存储
        for i in range(m):
            for j in range(n):
                if matrix[i][j]==0:
                    index.append((i, j))
        for i, j in index:
            # if i==0 and j==0:
            #     # 如果0在左上角
            #     matrix[0][1]=0
            #     matrix[1][0]=0
            # elif i==0 and j==n-1:
            #     # 如果0在右上角
            #     matrix[0][n-2]=0
            #     matrix[1][n-1]=0
            # elif i==m-1 and j==0:
            #     # 左下角
            #     matrix[m-1][1]=0
            #     matrix[m-2][0]=0
            # elif i==m-1 and j==n-1:
            #     # 右下角
            #     matrix[m-1][n-2]=0
            #     matrix[m-2][n-1]=0
            # # 下面讨论边的情况
            # elif i==0 and j>0 and j <n-1:
            #     # 第一行
            #     matrix[0][j-1]=0
            #     matrix[0][j+1]=0
            #     matrix[1][j]=0
            # elif i==m-1 and j>0 and j <n-1:
            #     # 最后一行
            #     matrix[m-1][j-1]=0
            #     matrix[m-1][j+1]=0
            #     matrix[m-2][j]=0
            # elif j==0 and i>0 and i<m-1:
            #     # 第一列
            #     matrix[i-1][0]=0
            #     matrix[i+1][0]=0
            #     matrix[i][1]=0
            # elif j==n-1 and i>0 and i<m-1:
            #     # 最后一列
            #     matrix[i-1][n-1]=0
            #     matrix[i+1][n-1]=0
            #     matrix[i][n-2]=0
            # elif i>0 and i<m-1 and j>0 and j<n-1:
            #     matrix[i][j+1] = 0
            #     matrix[i][j-1] = 0
            #     matrix[i-1][j] = 0
            #     matrix[i+1][j] = 0
            # 行置零
            matrix[i][:]=[0 for x in range(n)]
            # 列置零
            for i in range(m):
                matrix[i][j] = 0
# @lc code=end

