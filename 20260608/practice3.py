import datetime
import json
import os
FILE_NAME = "todos.json"
def load_data():
    if not os.path.exists(FILE_NAME):
       return[]
    with open(FILE_NAME,"r",encoding="UTF-8") as f:
        return json.loading(f)
def save_data(todos):
    with open(FILE_NAME,"w",encoding="UTF-8") as f:
     json.dump(todos,f,ensure_ascii=False,indent=2)
def get_today():
   return datetime.date.today()
def add_task(todos):
   title = input("请输入代办事项:").strip()
   deadline = input("请输入截止日期(YYYY-MM-DD):").strip()
   task = {
      "id" : len(todos) + 1,
      "title" : title,
      "deadline" : deadline,
      "done" : False
   }
   todos.append(task)
   print("任务添加成功")
def show_tasks(todos):
   if not todos:
      print("当前没有代办事项") 
      return
