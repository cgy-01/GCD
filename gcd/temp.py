# 定义文件路径
file_path = r"F:\Datasets\ChangeDetection\LEVIR-CD\list\train.txt"

# 打开文件，以写模式
with open(file_path, 'w') as file:
    # 生成文件名并写入文件
    for i in range(1, 446):
        file.write(f'train_{i}.png\n')
