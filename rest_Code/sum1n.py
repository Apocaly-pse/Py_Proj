# def sum1n(n):
#     try:
#         a = [0]
#         return a[n - 1] + n
#     except Exception:
#         return n + sum1n(n - 1)


def sum1n(n):
    if n == 1:
        return 1
    return sum1n(n - 1) + n


print(sum1n(10))
