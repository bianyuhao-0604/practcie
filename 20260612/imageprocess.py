import numpy as np
import matplotlib.pyplot as plt
img = plt.imread('D:/pythonprojects/practice-github/20260612/image/3.jpg')
R = img[:,:,0]
G = img[:,:,1]
B = img[:,:,2]
print("图像的 NumPy 数组形状:", img.shape)   # (5, 5, 3)
print("\nR 通道矩阵 (亮度值):")
print(R)
print("\nG 通道矩阵 (亮度值):")
print(G)
print("\nB 通道矩阵 (亮度值):")
print(B)
fig,axes = plt.subplots(1,4,figsize=(12,3))
axes[0].imshow(img)
axes[0].set_title("Original Image")
axes[1].imshow(R,cmap='gray')
axes[1].set_title("Red Channel")
axes[2].imshow(G,cmap='gray')
axes[2].set_title("Green Channel")
axes[3].imshow(B,cmap='gray')
axes[3].set_title("Blue Channel")
plt.tight_layout()
plt.show()

