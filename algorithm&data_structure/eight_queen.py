# coding=UTF-8
"""八皇后问题"""
# 判断下一个皇后是否发生冲突


# def conflict(state, nextX):
#     nextY = len(state)
#     for item in range(nextY):
#         # 判断对角线位置是否符合
#         if abs(nextX - state[item]) in [0, nextY - item]:
#             return True
#     return False


# def queens(num=8, state=()):
#     for pos in range(num):
#         if not conflict(state, pos):
#             if len(state) == num - 1:
#                 yield (pos, )
#             else:
#                 for result in queens(num, state + (pos,)):
#                     yield (pos,) + result


# def prettyprint(lst):
#     for i, item in enumerate(lst):
#         print("solution %d" % (i + 1))
#         for pos in item:
#             print('.' * pos + 'Q' + '.' * (len(item) - 1 - pos))
#         print()
#     print("Total solution: %d" % (i + 1))


# def solution(lst):
#     ret = []
#     for item in lst:
#         tmp = []
#         for pos in item:
#             tmp.append("." * pos + 'Q' + "." * (len(item) - 1 - pos))
#         ret.append(tmp)
#     return ret

class Solution:
    # 判断下一个皇后是否发生冲突
    def conflict(self, state, nextX):
        nextY = len(state)
        for i in range(nextY):
            # 判断对角线位置是否符合
            if abs(nextX - state[i]) in [0, nextY - i]:
                return True
        return False

    def queens(self, num, state=()):
        for pos in range(num):
            if not self.conflict(state, pos):
                if len(state) == num - 1:
                    yield (pos, )
                else:
                    for result in self.queens(num, state + (pos,)):
                        yield (pos,) + result

    def solveNQueens(self, n: int):
        ret = []
        for item in list(self.queens(n)):
            tmp = []
            for pos in item:
                tmp.append('.' * pos + 'Q' + '.' * (len(item) - 1 - pos))
            ret.append(tmp)
        return ret


s = Solution().solveNQueens(4)
print(s)

# N = 4
# print(list(queens(N)), "\n", len(list(queens(N))))
# prettyprint(list(queens(N)))
# print(solution(list(queens(N))))
