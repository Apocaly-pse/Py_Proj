#
# @lc app=leetcode.cn id=13 lang=python3
#
# [13] 罗马数字转整数
#

# @lc code=start
class Solution:
    def romanToInt(self, s: str) -> int:
        # 可以采用枚举法列出每一个字符串代表的数字并循环替换
        # 需要考虑7+6=13中情况(效率较低)
        # 这里采用数字的特征以及字典对应进行求解
        dic = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000
        }
        ret = 0
        for i in range(len(s)):
            if i < len(s) - 1 and dic[s[i]] < dic[s[i + 1]]:
                ret -= dic[s[i]]
            else:
                ret += dic[s[i]]
        return ret

# @lc code=end

