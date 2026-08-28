FILENAME = "students.txt"
def load_student():#文件加载函数
    students = {}
    try:
        with open(FILENAME,"r",encoding="UTF-8") as f:
            for line in f:
                name,score = line.strip().split(",")
                students[name] =  int(score)
    except FileNotFoundError:
        pass
    return students
def save_student(students):#数据保存函数
    with open(FILENAME,"w",encoding="UTF-8") as f:
        for name,score in students.items():
            f.write(f"{name},{score}\n")
def add_student(students):#添加数据函数
    name = input("请输入添加学生姓名:").strip()
    if name in students:
        print("该学生已存在")
        return 
    score = int(input("请输入学生成绩:,"))
    students[name] = score
    print("添加成功")
def show_all(students):#展示所有学生
    if not students:
        print("暂无学生数据")
        return
    print("\n打印所有学生成绩:")
    for name,score in students.items():
        print(f"姓名:{name:<10},成绩:{score}")
    print()
def search_student(students):#查询学生
    name = input("请输入查询学生姓名:").strip()
    score = students.get(name)
    if score is None:
        print("未找到该学生")
    else:
        print(f"{name},的成绩是{score}")
def update_student(students):#修改学生
    name = input("请输入修改学生姓名:").strip()
    if name not in students:
        print("未找到该学生")
        return
    new_score = int(input("请输入新的成绩:"))
    students[name] = new_score
    print("修改成功")
def delete_student(students):
    name = input("请输入删除的学生姓名:").strip()
    if students.pop(name,None):
       print("删除成功")
    else:
       print("未找到该学生")
def statistics(students):
    if not students:
        print("空列表")
        return
    scores = list(students.values())
    print(f"学生人数:{len(students)}")
    print(f"平均分:{sum(scores)/len(scores):.2f}")
    print(f"最高分:{max(scores)}")
    print(f"最低分:{min(scores)}")
def menu():
    print("\n====== 学生成绩管理系统 ======")
    print("1. 添加学生")
    print("2. 查看所有学生")
    print("3. 查询学生")
    print("4. 修改成绩")
    print("5. 删除学生")
    print("6. 成绩统计")
    print("0. 退出系统")
    print("============================")
def main():
    students = load_student()
    while True:
        menu()
        choice = input("请输入选项:")
        if choice == "1":
            add_student(students)
        elif choice == "2":
            show_all(students)
        elif choice == "3":
            search_student(students)
        elif choice == "4":
            update_student(students)
        elif choice == "5":
            delete_student(students)
        elif choice == "6":
            statistics(students)
        elif choice == "0":
            save_student(students)
            print("数据已保存,再见")
            break
        else:
            print("无效输入,请重新选择")
if __name__ == "main":
    main()


          
