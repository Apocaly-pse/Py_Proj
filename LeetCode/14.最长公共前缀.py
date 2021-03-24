#
# @lc app=leetcode.cn id=14 lang=python3
#
# [14] 最长公共前缀
#

# @lc code=start
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        l = min([len(i) for i in strs])
        n = len(strs)
        ret = ''
        for i in range(n-1):
            for j in range(l):
                if strs[i][j] != strs[i+1][j]:
                    break
                else:
                    ret += strs[i][j]
                    for k in range(len(ret)):
                        if ret[k] != strs[i+1][k]:
                            break                            
                        
        return ret
                    
# @lc code=end

