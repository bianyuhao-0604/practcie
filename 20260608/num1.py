import numpy as np
matrix =np.arange(50).reshape(10,5)
print("原始矩阵(10行x5列):")
print(matrix)
print("-"*50)

element = matrix[2,1]
print("练习1结果:")
print("第3行,第2列的元素是:",element)
print("-"*50)

row_5 = matrix[4,:]
print("练习2结果(获取行):")
print("第5行的完整数据",row_5)
print("-"*50)

col_3 = matrix[:,2]
print("练习3结果(获取列):")
print("第3列的完整数据",col_3)
print("-"*50)

rows_5_to_8 = matrix[4:8,:]
print("练习4结果:")
print("第5行到第8行的子矩阵:",rows_5_to_8)
print("-"*50)

big_matrix = np.arange(250).reshape(10,25)
print("为了练习 4 创建的更大矩阵 (10行 x 25列),前5行预览:")
print(big_matrix[:5,:])
print("\n")

cols_4_to_20 = big_matrix[:,3:20]
print("练习 4 结果:")
print("从大矩阵中提取的第 4 列到第 20 列的子矩阵 (10行 x 17列) 的形状:",cols_4_to_20.shape)
print(cols_4_to_20[:3, :])