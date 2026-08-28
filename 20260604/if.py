print("欢迎使用学生成绩评价系统")
try:
    score=float(input("请输入成绩(0-100):"))
except ValueError:
    print("输入错误,请输入有效数字(0-100)")
    exit()
if score <0 or score >100:
    print("成绩范围错误,请重新输入。")
else:
    if score >=90:
       grade = 'A'
       if score ==100:
           comment = '满分,太厉害啦!'
       else:
           comment = '优秀,你真棒!'
    elif score >=80:
       grade = 'B'
       comment = '良好,继续努力!'
    elif score >= 70:
       grade = 'C'
       comment = '中等,要加油了!'
    elif score >= 60:
       grade = 'D'
       comment = '及格,进步空间很大。'
    else:
       grade = 'F'
       comment = '不及格,需要补考。'
print("\n评价结果")   
print(f"等级:{grade}")
print(f"分数:{score:.1f}")
print(f"评价:{comment}")
print("-"*20)

       
    