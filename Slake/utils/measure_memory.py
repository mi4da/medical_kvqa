import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, ViTFeatureExtractor, ViTModel
from torchvision import transforms
import timm
from torchsummary import summary
from PIL import Image
import numpy as np
def measure_memory_usage(model, input_data):
    batch_sizes = [1, 2, 4]  # 不同的批处理大小
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for batch_size in batch_sizes:
        input_data_batched = input_data.repeat(batch_size, 1, 1, 1).to(device)  # 复制输入以匹配不同批处理大小
        model = model.to(device)

        # 使用 torch.cuda.memory_allocated() 来测量显存占用量
        torch.cuda.reset_peak_memory_stats()  # 重置显存统计
        _ = model(input_data_batched)  # 模拟前向传播
        memory_usage = torch.cuda.memory_allocated(device)

        print(f'Batch Size: {batch_size}, Memory Usage: {memory_usage / (1024 ** 2):.2f} MB')

if __name__ == "__main__":
    # 使用 ViTFeatureExtractor 加载预训练模型
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTModel.from_pretrained('google/vit-base-patch16-224').train()


    # model = timm.create_model(
    #     'maxvit_base_tf_384.in21k_ft_in1k',
    #     pretrained=True,
    #     num_classes=0,  # remove classifier nn.Linear
    # )
    # data_config = timm.data.resolve_model_data_config(model)
    # transforms = timm.data.create_transform(**data_config, is_training=False)

    # 创建一个随机的224x224x3的NumPy数组
    random_image = np.random.randint(0, 256, size=(384, 384, 3), dtype=np.uint8)

    # 将NumPy数组转换为PIL图像对象
    pil_image = Image.fromarray(random_image)

    # 使用 feature_extractor 对输入数据进行编码
    # inputs = transforms(pil_image)
    inputs = feature_extractor(pil_image)

    measure_memory_usage(model, torch.tensor(inputs['pixel_values'][0]))


    """
    maxvit
    ssh://root@172.16.1.144:22/root/anaconda3/envs/CrossModalSearch/bin/python -u /home/yc/cross-modal-search-demo/datasets/Slake/utils/measure_memory.py
Batch Size: 1, Memory Usage: 2912.37 MB
Batch Size: 2, Memory Usage: 5317.95 MB
Batch Size: 4, Memory Usage: 10153.44 MB
    vit
    Batch Size: 1, Memory Usage: 474.72 MB
    Batch Size: 2, Memory Usage: 615.15 MB
    Batch Size: 4, Memory Usage: 875.50 MB
    """