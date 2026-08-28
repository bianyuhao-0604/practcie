import numpy as np
import matplotlib.pyplot as plt
img_path = r'D:\pythonprojects\practice-github\20260612\image\3.jpg'
try:
    img = plt.imread(img_path)
except FileNotFoundError:
    print(f"错误:找不到文件'{img_path}'")
    exit()
img_float = img.astype(np.float32)
def normalize_minmax_01(img):
    min_val = img.min()
    max_val = img.max()
    if max_val == min_val:
        return np.zeros_like(img)
    return(img - min_val) / (max_val - min_val)
img_norm_01 = normalize_minmax_01(img_float)
# --- 新增：查看归一化后的数值 ---
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
#fig,axes = plt.subplots(1,2,figsize=(12,6))
#axes[0].imshow(img)
#axes[0].set_title("Original image(0-255)")
#axes[0].axis('off')
#axes[1].imshow(img_norm_01)
#axes[1].set_title("Normalized Image [0,1]")
#axes[1].axis('off')
# plt.tight_layout()
#plt.show()
if img.ndim == 2:
    channel_names = ['Gray']
    colors = ['gray']
    imgs_to_plot = [img]
    imgs_norm_to_plot = [img_norm_01]
elif img.ndim == 3 and img.shape[2] == 3:
    channel_names = ['Red','Green','Blue']
    colors = ['red','green','blue']
    imgs_to_plot = [img[:,:,0],img[:,:,1],img[:,:,2]]
    imgs_norm_to_plot = [img_norm_01[:,:,0],img_norm_01[:,:,1],img_norm_01[:,:,2]]
else:
    print("不支持的图像格式")
    exit()
if img.ndim == 3:
    fig = plt.figure(figsize=(18,10))
    gs = fig.add_gridspec(2,3)
else:
    fig = plt.figure(figsize=(12,10))
    gs = fig.add_gridspec(2,2)
ax_orig = fig.add_subplot(gs[0,0])
ax_orig.imshow(img)
ax_orig.set_title("Original Image(0-255)")
ax_orig.axis('off')
ax_norm = fig.add_subplot(gs[0,1])
ax_norm.imshow(img_norm_01)
ax_norm.set_title("Normalized Image [0,1]")
ax_norm.axis('off')
ax_hist = fig.add_subplot(gs[1,:])
for i in range(len(channel_names)):
    ax_hist.hist(imgs_to_plot[i].flatten,bins = 64,range =(0,255),
                color = colors[i],alpha=0.5,label=f'{channel_names[i]} (Original)')
    ax_hist.hist(imgs_norm_to_plot[i].flatten(), bins=64, range=(0, 1), 
                   color=colors[i], alpha=0.5, histtype='step', linewidth=2, label=f'{channel_names[i]} (Normalized)')
ax_hist.set_xlabel('Pixel Value')
ax_hist.set_ylabel('Frequency')
ax_hist.set_title('Histogram Comparison (Original vs Normalized)')
ax_hist.legend()
plt.tight_layout()
plt.show()  