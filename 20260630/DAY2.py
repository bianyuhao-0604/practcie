keyword = 'py2026'
count = 3
while count > 0:
    a = input(f'请输入密码(剩余次数为{count}次):')
    if a == keyword:
        print('密码正确')
        break
    count -= 1
else:
    print('账户已锁定')
    