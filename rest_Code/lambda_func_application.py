# 待排序的列表
lst = [
    {"name": "xiaohong", "age": 18},
    {"name": "xiaoli", "age": 12},
    {"name": "xiaoming", "age": 28},
    {"name": "xiaoliu", "age": 19},
]

# 此列表不能通过列表内置的排序命令（sort）进行排序
# lst.sort()
# TypeError: '<' not supported between instances of 'dict' and 'dict'


# 此时可以采用匿名函数的方法，读取键的方式实现排序
# lst.sort(key=lambda x: x["name"])
lst.sort(key=lambda x: x["age"])

print(lst)
