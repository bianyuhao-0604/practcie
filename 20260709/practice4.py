def add(a:float,b:float) -> float:
    return a + b
def multiply(a:float,b:float) -> float:
    return a * b
def _main():
    import sys
    if len(sys.argv) != 4:
        print("用法: python calculator.py <操作> <数字1> <数字2>")
        print("操作: add, multiply")
        sys.exit(1)
    op,a,b = sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
    if op == 'add':
        print(f"{a} + {b} = {add(a, b)}")
    elif op == 'multiply':
        print(f"{a} × {b} = {multiply(a, b)}")
    else:
        print(f"未知操作: {op}")
if __name__ == '__main__':
    _main()
