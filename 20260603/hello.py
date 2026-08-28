def set_difficulty():
    print("请选择游戏难度：")
    print("1.简单")
    print("2.普通")
    print("3.困难")
    while True:
     choice = input("请输入难度(1/2/3):").strip()
      if choice =="1":
        return 50,None
      elif choice=="2":
        return 100,20
      elif choice=="3":
        return 200,15
      else:
        print("无效输出,请重新输入。")
    

    