import os
import shutil

def count_parameters(model):
    return round(sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2))

def recreate_directory(directory_path):
    """
    如果目录存在，删除它及其所有内容，然后重新创建它。

    参数:
    - directory_path (str): 要重新创建的目录的路径。
    """

    # 检查目录是否存在
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        # 删除目录及其所有内容
        shutil.rmtree(directory_path)

    # 创建目录
    os.makedirs(directory_path)