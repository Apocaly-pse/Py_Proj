def convertToTitle(n: int) -> str:
    col = {}
    # 首先定义字典存储对应关系
    for i in range(26):
        col[i] = chr(65 + i)

    if n <= 26:
        return col[n]
    elif n <= 26**2 + 26:
        return col[(n-1)//26-1] + col[(n-1)%26]
    elif n <= 26**3 + 26:
        return col[n // 26 % 26] + col[n // 26 // 26] + col[n % 26]
    elif n <= 26**4 + 26:
        return col[n // 26 // 26 % 26] + col[n // 26 // 26 // 26] + col[n // 26 % 26] + col[n % 26]
    elif n <= 26**5 + 26:
        return col[n // 26 // 26 // 26 % 26] + col[n // 26 // 26 // 26 // 26] + col[n // 26 // 26 % 26] + col[n // 26 % 26] + col[n % 26]
    elif n <= 26**6 + 26:
        return col[n // 26 // 26 // 26 // 26 % 26] + col[n // 26 // 26 // 26 // 26 // 26%26] + col[n // 26 // 26 // 26 % 26] + col[n // 26 // 26 % 26] + col[n // 26 % 26] + col[n % 26]
    elif n <= 26**7 + 26:
        return col[n // 26 // 26 // 26 // 26 // 26 // 26%26] + col[n // 26 // 26 // 26 // 26 // 26 % 26] + col[n // 26 // 26 // 26 // 26 % 26] + col[n // 26 // 26 // 26 % 26] + col[n // 26 // 26 % 26] + col[n // 26 % 26] + col[n % 26]


n = 52
print(convertToTitle(n))
