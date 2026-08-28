import numpy as np
import matplotlib.pyplot as plt

# 图像路径
img_path = r'D:\pythonprojects\practice-github\20260612\image\3.jpg'

try:
    # 1. 读取图像
    img = plt.imread(img_path)
except FileNotFoundError:
    print(f"错误: 找不到文件 '{img_path}'")
    exit()

# 2. 转换为 float32 类型
img_float = img.astype(np.float32)

# 3. 定义并应用归一化函数
def normalize_minmax_01(img):
    """将图像像素值归一化到 [0, 1] 范围"""
    min_val = img.min()
    max_val = img.max()
    if max_val == min_val:
        return np.zeros_like(img)
    return (img - min_val) / (max_val - min_val)

img_norm_01 = normalize_minmax_01(img_float)

# --- 查看归一化后的数值 ---
print("--- 归一化后图像数据信息 ---")
print(f"数据类型: {img_norm_01.dtype}")
print(f"数值范围: 最小值 = {img_norm_01.min():.4f}, 最大值 = {img_norm_01.max():.4f}")
# 打印第一个通道（例如R通道）的前5x5像素块，方便直观查看数值
if img_norm_01.ndim == 3: # 如果是彩色图像
    print("\n示例数据 (归一化后图像的第一个通道，左上角 5x5 像素块):")
    print(img_norm_01[0:5, 0:5, 0])
else: # 如果是灰度图像
    print("\n示例数据 (归一化后图像，左上角 5x5 像素块):")
    print(img_norm_01[0:5, 0:5])
print("---------------------------------\n")

# --- 新增：绘制直方图对比 ---

# 确定图像类型并设置颜色
if img.ndim == 2: # 灰度图像
    channel_names = ['Gray']
    colors = ['gray']
    imgs_to_plot = [img]
    imgs_norm_to_plot = [img_norm_01]
elif img.ndim == 3 and img.shape[2] == 3: # 彩色图像 (RGB)
    channel_names = ['Red', 'Green', 'Blue']
    colors = ['red', 'green', 'blue']
    # 分离RGB通道
    imgs_to_plot = [img[:, :, 0], img[:, :, 1], img[:, :, 2]]
    imgs_norm_to_plot = [img_norm_01[:, :, 0], img_norm_01[:, :, 1], img_norm_01[:, :, 2]]
else:
    print("不支持的图像格式。")
    exit()

# 创建一个大的画布，用于放置所有子图
# 如果是彩色图，我们创建2行3列的子图 (原始图像, 归一化图像, 直方图对比)
# 如果是灰度图，我们创建2行2列的子图
if img.ndim == 3:
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)
else:
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2)

# 第一行：显示原始图像和归一化图像
ax_orig = fig.add_subplot(gs[0, 0])
ax_orig.imshow(img)
ax_orig.set_title("Original Image (0-255)")
ax_orig.axis('off')

ax_norm = fig.add_subplot(gs[0, 1])
ax_norm.imshow(img_norm_01)
ax_norm.set_title("Normalized Image [0, 1]")
ax_norm.axis('off')

# 第二行：绘制直方图对比
ax_hist = fig.add_subplot(gs[1, :])

# 为每个通道绘制直方图
for i in range(len(channel_names)):
    # 原始图像的直方图 (0-255)
    ax_hist.hist(imgs_to_plot[i].flatten(), bins=64, range=(0, 255), 
                   color=colors[i], alpha=0.5, label=f'{channel_names[i]} (Original)')
    
    # 归一化后图像的直方图 (0-1)
    ax_hist.hist(imgs_norm_to_plot[i].flatten(), bins=64, range=(0, 1), 
                   color=colors[i], alpha=0.5, histtype='step', linewidth=2, label=f'{channel_names[i]} (Normalized)')

ax_hist.set_xlabel('Pixel Value')
ax_hist.set_ylabel('Frequency')
ax_hist.set_title('Histogram Comparison (Original vs Normalized)')
ax_hist.legend()

plt.tight_layout()
plt.show()