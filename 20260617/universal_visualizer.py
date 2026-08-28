import os
import json
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import hashlib

def get_color_for_label(label):
    """根据标签名生成固定的 RGB 颜色 (0-255)"""
    hash_val = int(hashlib.md5(label.encode()).hexdigest(), 16)
    r = (hash_val & 0xFF0000) >> 16
    g = (hash_val & 0x00FF00) >> 8
    b = hash_val & 0x0000FF
    return (r, g, b)

def visualize_bounding_boxes(image_path, bbox_file_path, format_type=None, class_names=None):
    """
    🌟 通用边界框可视化函数 (纯净边框 + 底部图例版)
    """
    # ================= 1. 加载图像 =================
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    draw = ImageDraw.Draw(img) # 注意：去掉了 "RGBA"，因为我们不再需要半透明文本框
    
    # 计算精致的边框粗细 (限制在 1px 到 4px 之间)
    min_dim = min(img_w, img_h)
    line_width = max(1, min(4, int(min_dim / 300)))

    # ================= 2. 自动推断文件格式 =================
    if format_type is None:
        ext = os.path.splitext(bbox_file_path)[1].lower()
        if ext == '.txt':
            format_type = 'yolo'
        elif ext == '.xml':
            format_type = 'voc'
        elif ext == '.json':
            with open(bbox_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict) and 'annotations' in data and 'images' in data:
                format_type = 'coco'
            elif isinstance(data, list):
                format_type = 'ls_json'
            else:
                raise ValueError("❌ 无法识别的 JSON 结构。")
    
    # ================= 3. 解析边界框坐标 =================
    boxes = []
    img_name = os.path.basename(image_path)
    
    if format_type == 'yolo':
        with open(bbox_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    xmin = int((cx - w / 2) * img_w)
                    ymin = int((cy - h / 2) * img_h)
                    xmax = int((cx + w / 2) * img_w)
                    ymax = int((cy + h / 2) * img_h)
                    label = class_names[cls_id] if class_names and cls_id < len(class_names) else f"class_{cls_id}"
                    boxes.append({'label': label, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
                    
    elif format_type == 'voc':
        tree = ET.parse(bbox_file_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            label = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            boxes.append({'label': label, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
            
    elif format_type == 'coco':
        with open(bbox_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cat_map = {c['id']: c['name'] for c in data.get('categories', [])}
        target_img_id = None
        for img_info in data.get('images', []):
            if img_info['file_name'] == img_name or img_info['file_name'].endswith(img_name):
                target_img_id = img_info['id']
                break
        if target_img_id is not None:
            for ann in data.get('annotations', []):
                if ann['image_id'] == target_img_id:
                    x, y, w, h = ann['bbox']
                    label = cat_map.get(ann['category_id'], 'unknown')
                    boxes.append({'label': label, 'xmin': int(x), 'ymin': int(y), 'xmax': int(x+w), 'ymax': int(y+h)})
                    
    elif format_type == 'ls_json':
        with open(bbox_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for task in data:
            task_img = os.path.basename(task.get('data', {}).get('image', ''))
            if task_img == img_name or task_img.endswith(img_name):
                for ann in task.get('annotations', []):
                    for res in ann.get('result', []):
                        v = res.get('value', {})
                        labels = v.get('rectanglelabels') or v.get('labels') or ['unknown']
                        x_pct, y_pct, w_pct, h_pct = v.get('x', 0), v.get('y', 0), v.get('width', 0), v.get('height', 0)
                        boxes.append({
                            'label': labels[0],
                            'xmin': int((x_pct / 100) * img_w),
                            'ymin': int((y_pct / 100) * img_h),
                            'xmax': int(((x_pct + w_pct) / 100) * img_w),
                            'ymax': int(((y_pct + h_pct) / 100) * img_h)
                        })

    # ================= 4. 绘制纯净边界框 (无文本) =================
    print(f"🐶 图片: {img_name} | 检测到 {len(boxes)} 个边界框")
    
    # 记录当前图片中实际出现的类别及其颜色，用于生成图例
    active_labels = {} 
    
    for box in boxes:
        label = box['label']
        xmin, ymin, xmax, ymax = box['xmin'], box['ymin'], box['xmax'], box['ymax']
        
        if label not in active_labels:
            active_labels[label] = get_color_for_label(label)
            
        color = active_labels[label]
        
        # 仅绘制边框，不绘制任何文字和背景块
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=line_width)

    # ================= 5. 使用 Matplotlib 显示图片并添加底部图例 =================
    # 设置中文字体，确保图例中的中文标签不乱码
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(np.array(img))
    ax.set_title(f"标注结果: {img_name}", fontsize=14)
    ax.axis('off') # 隐藏坐标轴
    
    # 🌟 核心优化：构建底部图例 (Legend)
    if active_labels:
        # 将 PIL 的 RGB (0-255) 转换为 Matplotlib 需要的 RGB (0.0-1.0)
        legend_handles = [
            mpatches.Patch(color=np.array(c) / 255.0, label=l) 
            for l, c in active_labels.items()
        ]
        
        # 将图例放置在图片正下方 (bbox_to_anchor 控制位置)
        ax.legend(
            handles=legend_handles,
            loc='upper center',          # 图例内部的锚点
            bbox_to_anchor=(0.5, -0.02), # 相对于 Axes 的位置 (0.5 是水平居中, -0.02 是稍微在图片下方)
            ncol=len(active_labels),     # 强制所有类别在同一行显示 (如果类别很多会自动适应)
            frameon=False,               # 去掉图例的白色背景框，更清爽
            fontsize=12,                 # 图例文字大小
            handlelength=1.5,            # 颜色色块的长度
            handletextpad=0.5            # 色块与文字的距离
        )
        
    # 调整布局，确保底部的图例不会被裁切掉
    plt.subplots_adjust(bottom=0.15) 
    plt.tight_layout()
    plt.show()

# ================= 测试用例 =================
if __name__ == "__main__":
    # 取消注释下面的代码来测试你的图片
    visualize_bounding_boxes("dogs/beagle/beagle_163.jpg", "JSON.json", class_names=['dog'])
    pass
