def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left,right)
def _merge(left,right):
     merged = []
     i = j = 0
     while i < len(left) and j < len(right):
        if left[i] < right[j]:
             merged.append(left[i])
             i += 1
        else:
            merged.append(right[j])
            j += 1
     merged.extend(left[i:])
     merged.extend(right[j:])
     return merged
if __name__ == "__main__":
    test_data = [38, 27, 43, 3, 9, 82, 10]
    sorted_result = merge_sort(test_data)
    print("O(n log n) Sorted Result:", sorted_result)