import os#导入os模块，实现与操作系统交互
def generate_content(num_lines):#定义一个函数
    content_lines = []#生成一个空列表
    for i in range(1,num_lines+1):#遍历从1至最后1行
       line = ' '.join([chr(j)*i for j in range(ord('A'),ord('Z')+1)])#ord函数字符转整数；chr函数整数转字符；join函数连接
       content_lines.append(line)#生成的行添加到列表末尾
    return content_lines#返回列表
root_path = r"D:\pythonprojects\practice-github\20260610\Data"#创建路径
num_folders = 10#创建文件夹
num_files_per_folder = 10#每个文件夹中创建文件
num_lines_per_file = 10#每个文件中创建行
os.makedirs(root_path, exist_ok = True)#确保根目录是存在的，不存在则自动创建
for folder_idx in range(1,num_folders + 1):#启动计数循环，从1至num_folders
    folder_name = f"Data_{folder_idx:04d}"#生成文件夹名称
    folder_path = os.path.join(root_path,folder_name) #将文件夹名称与根目录安全拼接
    os.makedirs(folder_path, exist_ok = True)#确保文件夹路径存在，不存在则自动创建
    for file_idx in range(1,num_files_per_folder+1):#启动计数循环，从1至num_files_per_folder
        file_name = f"Data_{folder_idx:04d}_{file_idx:02d}.txt"#生成文件名
        file_path = os.path.join(folder_path,file_name)#文件名与文件路径拼接
        lines = generate_content(num_lines_per_file)#生成文件内容
        with open(file_path,'w',encoding='UTF-8') as f:#写入文件
            f.write('\n'.join(lines))
print(f"任务完成！已在 {root_path} 下创建 {num_folders} 个子文件夹，每个子文件夹包含 {num_files_per_folder} 个文件。")