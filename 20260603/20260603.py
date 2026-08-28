import random
print("【欢迎来到猜数字游戏!】")
target_number = random.randint(1,100)
attempts=0


while True:
    attempts += 1
    try:
       guess=int(input(f"第{attempts}次猜数,请输入一个1-100的数字(输入0退出):"))
    except ValueError:
       print("输入错误，请输入一个整数。")
       continue
    if guess==0:
       print("游戏结束!")
       break
     
    if guess < target_number:
       print("太小啦，再猜一猜.")
    elif guess > target_number:
       print("太大啦，再猜一猜.")
    else:
       print("恭喜你，猜对啦!")
       print(f"一共猜了{attempts}次")         
       break

