balance=1000.0
def check_balance():#定义一个余额查询函数
    print(f"您的账户余额为,￥{balance:.2f}")
def deposit(amount):#定义函数：存款
    global balance
    if amount > 0:
        balance += amount
        print(f"存款成功,￥{balance:.2f}") 
        check_balance()
    else:
        print('存款失败,输入大于0的金额。')
def withdraw(amount):#定义函数：取款
    global balance
    if amount <= 0:
        print('取款失败,请输入大于0的金额。')
    elif amount > balance:
        print('取款失败,余额不足。')
    else:
        balance -=amount
        print(f"取款成功,您取出了,￥{amount:.2f}。")
        check_balance()
def transfer(amount):
    global balance
    if amount <=0:
        print('转账失败,请输入大于0的数。')
    elif amount > balance:
        print('转账失败,余额不足。')
    else:
        balance -=amount
        print(f"转账成功,您转出了,￥{amount:.2f}。")
        check_balance()
def show_menu():#定义主菜单
    print("\n-----简易银行系统-----")
    print('1.查询余额')
    print('2.存款')
    print('3.取款')
    print('4.转账')
    print('5.退出')
    print('-'*20)
print('欢迎使用简易银行系统!')#主循环
while True:
    show_menu()
    choice = input("请输入操作(1-5)")
    if choice == "1":
       try:
        check_balance()
       except:
           print('输入无效,请输入数字。')
    elif choice == "2":
       try:
         amount = float(input('请输入您的存款金额，'))
         deposit(amount)
       except:
           print('输入无效,请输入数字。')
    elif choice == "3":
       try:  
        amount = float(input('请输入您的取款金额，'))
        withdraw(amount)
       except:
           print('输入无效,请输入数字。')
    elif choice == "4":
       try:
        amount = float(input('请输入您的转账金额，'))
        transfer(amount)
       except:
           print('输入无效,请输入数字。')
    elif choice == "5":
        print('感谢您的使用，再见!')
        break
    else:
        print('无效操作,请重新输入。')
