# fibonacci number recursion algorithm
def fib_recursion(n):
    if n < 2:
        return n
    return fib_recursion(n - 1) + fib_recursion(n - 2)


print(fib_recursion(20))
