#
# @lc app=leetcode.cn id=500 lang=python3
#
# [500] 键盘行
#

# @lc code=start
class Solution:
    def findWords(self, words: List[str]) -> List[str]:
        s1, s2, s3 = "qwertyuiop", "asdfghjkl", "zxcvbnm"
        ret = []
        for word in words:
            if set(word) in set(s1 + s1.upper()):
                ret.append(word)
            elif set(word) in set(s2 + s2.upper()):
                ret.append(word)
            elif set(word) in set(s3 + s3.upper()):
                ret.append(word)
        return ret

# @lc code=end
