#
# @lc app=leetcode.cn id=66 lang=python3
#
# [66] 加一
#

# @lc code=start
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        a=''
        b=[]
        for i in digits:
            a+=str(i)
        for j in list(str(int(a)+1)):
            b.append(int(j))
        return b
# @lc code=end

