# 使用辗转相除法计算两数的最大公因数（循环实现）
def gcd_loop(m, n):
    while m % n != 0:
        m, n = n, m % n
    return n


print(gcd_loop(20, 36))
