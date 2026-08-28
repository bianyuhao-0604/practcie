from nilearn import plotting
import nibabel as nib

file_path = 'test3.mgh'
img = nib.load(file_path)

# 如果是 4D fMRI 数据，使用 mean_img 计算时间平均图，或 index_img 提取特定帧
if img.get_fdata().ndim == 4:
    from nilearn.image import mean_img
    img_to_plot = mean_img(img)
else:
    img_to_plot = img

# 绘制解剖/功能图像 (自动寻找最佳切片位置)
# display_mode='ortho' 同时显示三个正交面
display = plotting.plot_epi(img_to_plot, 
                            display_mode='ortho', 
                            title='fMRI EPI / Anatomy',
                            cmap='gray', # 如果是统计激活图，可改为 'hot' 或 'cold_hot'
                            cut_coords=(0, -20, 10)) # 可手动指定 MNI 或原生坐标系切片位置

# 保存为高清图片
display.savefig('mgh_visualization.png', dpi=300)
plotting.show()