def sum_(n):
    if n < 2:
        return 1
    return sum_(n - 1) + n


print(sum_(100))
