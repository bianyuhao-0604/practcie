import matplotlib.pyplot as plt

# --- 解决中文显示问题的核心设置 ---
# 方法A：使用黑体 (SimHei)
plt.rcParams['font.sans-serif'] = ['SimHei'] 

# 方法B：使用微软雅黑 (Microsoft YaHei)
# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']

# 方法C：使用宋体 (SimSun)
# plt.rcParams['font.sans-serif'] = ['SimSun']
# -------------------------------------

# 解决坐标轴负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False

# 现在您可以正常绘制包含中文的图表了
plt.title('这是一个中文标题')
plt.xlabel('x轴标签')
plt.ylabel('y轴标签')
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()