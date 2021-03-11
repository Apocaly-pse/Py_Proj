# encoding=utf-8
"""
本程序为Python编写的有理数类
"""


class Rational(object):
    def _gcd(m, n):
        pass

    def __init__(self, num, den=1):
        # num=numerator为分子
        # den=denominator为分母
        self.num = num
        self.den = den

    def plus(self, another):
        den = self.den * another.den
        num = (self.num * another.den + self.den * another.num)
        # 相加后返回一个新的有理数类
        return Rational(num, den)

    def print_(self):
        print(str(self.num) + "/" + str(self.den))


x = Rational(3, 5)
y = Rational(4, 6)

z = x.plus(y)
# z.print_()
print(type(z))
