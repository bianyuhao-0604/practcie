def reverse_string(s:str) -> str:
    """反转字符串"""
    return s[::-1]
def count_vowels(s:str) -> int:
    """统计元音字母个数"""
    return sum(1 for c in s.lower() if c in 'aeiou')
print(f'模块 __name__ = {__name__}')
if __name__ == '__main__':
    print("🧪 运行模块自测...")
    assert reverse_string("hello") == "olleh"
    assert count_vowels("Python") == 1
    print("✅ 所有测试通过！")
