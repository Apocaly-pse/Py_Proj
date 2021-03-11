# calc gcd of two natural nuaber by recursion
def gcd_recur_1(a, b):
    return gcd_recur_1(b, a % b) if a % b else b


def gcd_recur_2(a, b):
    if a % b == 0:
        return b
    return gcd_recur_2(b, a % b)


def gcd_recur_3(a, b):
    # return a if not b else gcd_recur_3(b, a % b)
    return a if b == 0 else gcd_recur_3(b, a % b)

print(gcd_recur_1(60, 24))
print(gcd_recur_2(60, 24))
print(gcd_recur_3(60, 24))
