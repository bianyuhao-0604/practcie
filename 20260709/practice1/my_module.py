print("🔄 my_module 被加载了！")
COUNT = 0
def increment():
    global COUNT
    COUNT += 1
    return COUNT