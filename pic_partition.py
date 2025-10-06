#根据前面划分的标签集将图片划分到指定的文件夹中
import os
import shutil

# 原始图片所在文件夹（所有图片都存放在这里）
src_image_dir = r".\CelebA\Img\img_align_celeba"

# 需要处理的 txt 文件列表（可同时处理训练/验证/测试集的分配）
txt_file_paths = [
    r".\CelebA\Img\train_txt.txt",  # 训练集分配文件
    r".\CelebA\Img\val_txt.txt",  # 验证集分配文件
    r".\CelebA\Img\test_txt.txt"  # 测试集分配文件
]


# 分配图片的函数
def distribute_images(txt_path):
    """
    根据单个 txt 文件的路径信息，将图片复制到目标文件夹
    :param txt_path: 包含“目标路径 标签”的 txt 文件路径
    """
    with open(txt_path, "r") as f:
        lines = f.readlines()  # 读取所有行

    for line in lines:
        line = line.strip()  # 去除行首尾空格
        if not line:  # 跳过空行
            continue

        parts = line.split()  # 按空格分割行内容
        if len(parts) < 1:  # 跳过格式错误的行
            continue

        target_path = parts[0]  # 提取目标路径（如 ".\CelebA\Img\Test\12345.jpg"）
        file_name = os.path.basename(target_path)  # 提取文件名（如 "12345.jpg"）
        src_file_path = os.path.join(src_image_dir, file_name)  # 原始图片的完整路径

        # 确保目标文件夹存在（不存在则自动创建）
        target_dir = os.path.dirname(target_path)
        os.makedirs(target_dir, exist_ok=True)

        # 复制图片到目标路径（如需“移动”而非“复制”，可替换为 shutil.move）
        try:
            shutil.copy(src_file_path, target_path)
            print(f"成功：{src_file_path} → {target_path}")
        except FileNotFoundError:
            print(f"警告：源文件不存在 → {src_file_path}")
        except Exception as e:
            print(f"错误：{e}")

if __name__ == '__main__':
    # 批量处理所有 txt 文件,生成对应的训练、测试数据集
    for txt_path in txt_file_paths:
        distribute_images(txt_path)