import os
import shutil
import pandas as pd

# 设置输入和输出路径
input_path = "/gpfs/workdir/islamm/datasets"
output_path = "/gpfs/workdir/islamm/datasets_without_NoFindings"
csv_file_path = "/gpfs/workdir/islamm/datasets/Data_Entry_2017.csv"

# 创建输出目录（如果不存在）
os.makedirs(output_path, exist_ok=True)

# 读取CSV文件
df = pd.read_csv(csv_file_path)

# 获取“Finding Labels”是"No Finding"的图片索引
no_finding_images = df[df['Finding Labels'] == 'No Finding']['Image Index']

# 遍历 input_path 路径下的 images_001 到 images_012 文件夹及其子文件夹中的图片
for folder_name in range(1, 13):  # 从 images_001 到 images_012
    folder_path = os.path.join(input_path, f'images_{folder_name:03d}')
    
    if os.path.exists(folder_path):
        for root, dirs, files in os.walk(folder_path):  # 遍历该文件夹下的所有子文件夹
            for image_filename in files:
                if image_filename.endswith('.png'):
                    # 判断图片文件名是否在"No Finding"的列表中
                    if image_filename not in no_finding_images.values:
                        # 构造输入图片路径和输出图片路径
                        input_image_path = os.path.join(root, image_filename)
                        output_image_path = os.path.join(output_path, image_filename)
                        
                        # 将图片复制到输出路径
                        shutil.copy(input_image_path, output_image_path)

print("图片保存完毕")

