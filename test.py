def findMaxAverage(nums, k: int) -> float:
    ret = sum(nums[:k])
    for i in range(len(nums) - k):
        ret = max(ret - nums[i] + nums[i + k], ret)
    return ret / k


print(findMaxAverage([0, 4, 0, 3, 1], 1))
