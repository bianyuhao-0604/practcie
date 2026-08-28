class BankCard:
    def __init__(self,card_number,balance = 0.0):
        clean = card_number.replace(" ","")#去除卡号空格
        if not clean.isdigit() or len(clean) != 16:#判断是否是16位数字
            raise ValueError("卡号必须为16位数字,空格可选。")
        self.__card_number = clean
        self._balance = balance
    @property
    def balance(self):
        return self._balance
    @property
    def card_number(self):
        return f"**** **** **** {self.__card_number[-4:]}"
    def deposit(self,amount):
        if amount <= 0:
           print("存款金额不能小于零")
           return
        self._balance += amount
        print(f"存款成功,账户余额为,￥{self._balance:.2f}")
    def withdraw(self,amount):
       if amount <= 0:
          print("取款金额不能小于零")
          return
       if amount > self._balance:
          print("余额不足")
          return
       self._balance -=amount
       print(f"取款成功,账户余额为,￥{self._balance:.2f}")
    def transfer_out(self,target_card,amount):
      if not isinstance(target_card,BankCard):
         print("目标卡无效")
         return
      if amount > self._balance:
         print("余额不足")
         return
      if amount <= 0:
         print("请输入有效金额")
         return
      self._balance -=amount
      target_card._balance +=amount
      print(f"已向{target_card.card_number},转账￥{amount:.2f}。")
# 测试
card1: BankCard = BankCard("1234 5678 9012 3456", 1000)
card2 = BankCard("9876543210987654", 500)
print(card1.card_number)      # **** **** **** 3456
card1.deposit(500)            # 余额 1500
card1.withdraw(200)           # 余额 1300
card1.transfer_out(300, card2)
print(card1.balance)          # 1000.0
print(card2.balance)          # 800.0
     
