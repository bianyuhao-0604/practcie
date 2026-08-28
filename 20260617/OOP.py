class BankAccount:
    def __init__(self,account_number,owner,balance=0):
        self.__account_number = account_number
        self.__owner = owner
        self.__balance = balance
    def deposit(self,amount):
        self.__balance += amount
    def withdraw(self,amount):
        self.__balance -= amount