#
# @lc app=leetcode.cn id=14 lang=python3
#
# [14] 最长公共前缀
#

# @lc code=start
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        # 本题技巧:python自带的zip解包函数
        ret = ''
        for letter in zip(*strs):
            if len(set(letter)) == 1:
                # 如果每一个字母的开头作为一个集合仅有一个元素,说明其首字母是相同的
                ret += letter[0]
            else:
                # 如果不存在,则直接跳出循环,不再进行检索
                break
        return ret
# @lc code=end
