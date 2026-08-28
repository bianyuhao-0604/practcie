import os
import glob

def clean_garbage_prefixes(image_dir, label_dir, dry_run=True):
    """
    批量清洗图片和标签的乱码前缀。
    :param dry_run: True=只打印预览不修改，False=真实执行重命名
    """
    separators = ('_', '-', ' ') # 定义哪些符号被认为是“前缀”和“核心名”的分界线
    action = "👀 预览模式 (不会修改文件)" if dry_run else "🔥 真实修改模式 (正在重命名...)"
    print(f"🚀 启动清洗: {action}")
    print("-" * 50)
    
    # 收集所有需要处理的文件 (图片 + txt标签)
    all_files = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        all_files.extend(glob.glob(os.path.join(image_dir, ext)))
    all_files.extend(glob.glob(os.path.join(label_dir, '*.txt')))
        
    rename_count = 0
    conflict_count = 0
    skip_count = 0
    
    for file_path in all_files:
        dir_name = os.path.dirname(file_path)
        old_name = os.path.basename(file_path)
        ext = os.path.splitext(old_name)[1] # 获取后缀，如 .jpg 或 .txt
        
        # 提取核心名字 (去掉第一个分隔符之前的内容)
        base_name = os.path.splitext(old_name)[0]
        first_sep_idx = -1
        for sep in separators:
            idx = base_name.find(sep)
            # 找到最靠前的那个分隔符
            if idx != -1 and (first_sep_idx == -1 or idx < first_sep_idx):
                first_sep_idx = idx
                
        # 如果找到了分隔符，且分隔符不在文件名的最开头
        if first_sep_idx > 0:
            core_name = base_name[first_sep_idx + 1:] # 截取分隔符后面的所有内容
            new_name = core_name + ext
            new_path = os.path.join(dir_name, new_name)
            
            # 如果新名字和老名字不一样，才需要重命名
            if old_name != new_name:
                # 🛡️ 安全检查：防止重名覆盖已有的文件
                if not os.path.exists(new_path):
                    if not dry_run:
                        os.rename(file_path, new_path) # 真实重命名
                    print(f"  ✏️ {old_name}  ➡️  {new_name}")
                    rename_count += 1
                else:
                    print(f"  ⚠️ 冲突跳过: {old_name} (因为 {new_name} 已经存在了)")
                    conflict_count += 1
            else:
                skip_count += 1
        else:
            skip_count += 1 # 没有前缀，不需要改

    print("-" * 50)
    print(f"✅ 处理完毕！")
    print(f"   - 准备重命名: {rename_count} 个文件")
    print(f"   - 冲突跳过: {conflict_count} 个文件")
    print(f"   - 无需修改: {skip_count} 个文件")
    
    if dry_run:
        print("\n💡 【重要提示】: 当前是预览模式。如果上面的预览结果符合你的预期，")
        print("   请将代码底部的 dry_run=True 改为 dry_run=False，然后再次运行！")

# ================= 🌟 配置区域 (只需修改这里) =================
if __name__ == "__main__":
    # 1. 填入你的图片文件夹路径
    IMG_DIR = "dogs"   # 👈 改成你的真实路径，例如 "C:/my_data/images"
    
    # 2. 填入你的标签文件夹路径
    LBL_DIR = "YOLO/labels"   # 👈 改成你的真实路径，例如 "C:/my_data/labels"
    
    # 3. 第一次运行，请保持 True 不变！
    clean_garbage_prefixes(IMG_DIR, LBL_DIR, dry_run=False) 