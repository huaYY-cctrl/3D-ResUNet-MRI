import random
import torch
import numpy as np
from torch.utils.data import Dataset


class Image_Label_train(Dataset):
    """训练集数据加载器，支持多种数据增强操作"""
    def __init__(self, image_label_pairs):
        """
                初始化训练集数据加载器

                参数:
                    image_label_pairs: 图像和标签路径的元组列表，格式为[(image_path1, label_path1), (image_path2, label_path2), ...]
        """
        self.image_label_pairs = image_label_pairs

    def __getitem__(self, index):
        """获取单个数据样本，包含随机数据增强"""

        # 读取图像和标签数据
        image_path = self.image_label_pairs[index][0]
        label_path = self.image_label_pairs[index][1]
        image_array = np.float32(np.load(image_path))  # 加载图像并转换为float32类型
        label_array = np.float32(np.load(label_path))  # 加载标签并转换为float32类型
        # 数据有30%的概率不做增强
        if np.random.random_sample() > 0.7:
            # 不做增强，直接添加通道维度并转换为PyTorch张量
            image_data = np.expand_dims(image_array, axis=0) # 添加通道维度 (C=1, D, H, W)
            label_data = np.expand_dims(label_array, axis=0) # 添加通道维度
            return torch.from_numpy(image_data), torch.from_numpy(label_data)
        else:
            # 应用多种数据增强操作
            # 随机水平翻转 (概率20%)
            if np.random.random_sample() > 0.8:
                image_array = np.flip(image_array, axis=0).copy() # 沿深度方向翻转
                label_array = np.flip(label_array, axis=0).copy() # 确保标签与图像同步翻转
            # 随机垂直翻转 (概率20%)
            if np.random.random_sample() > 0.8:
                image_array = np.flip(image_array, axis=1).copy() # 沿高度方向翻转
                label_array = np.flip(label_array, axis=1).copy()
            # 随机前后翻转 (概率20%)
            if np.random.random_sample() > 0.8:
                image_array = np.flip(image_array, axis=2).copy() # 沿宽度方向翻转
                label_array = np.flip(label_array, axis=2).copy()
            # 随机旋转90度的倍数 (概率50%)
            if np.random.random_sample() > 0.5:
                k = np.random.randint(-3, 4) # 随机选择旋转次数 (-3到3次，对应-270到270度)
                image_array = np.rot90(image_array, k, axes=(1, 2)).copy() # 在H-W平面旋转
                label_array = np.rot90(label_array, k, axes=(1, 2)).copy() # 确保标签与图像同步旋转
            # 随机亮度缩放 (概率10%)
            if np.random.random_sample() > 0.9:
                scale = np.float32(np.random.uniform(low=0.9, high=1.1, size=1)) # 亮度缩放因子
                image_array = image_array * scale # 调整图像亮度
            # 随机添加高斯噪声 (概率10%)
            if np.random.random_sample() > 0.9:
                variance = random.uniform(0, 0.1) # 随机噪声方差
                image_array = image_array + np.random.normal(0.0, variance, image_array.shape).astype('float32') # 添加高斯噪声
            # 添加通道维度并转换为PyTorch张量
            image_data = np.expand_dims(image_array, axis=0)
            label_data = np.expand_dims(label_array, axis=0)
            return torch.from_numpy(image_data), torch.from_numpy(label_data)

    def __len__(self):
        return len(self.image_label_pairs)



class Image_Label_valid(Dataset):
    """验证集数据加载器，不进行数据增强"""
    def  __init__(self, image_label_pairs):
        """
                初始化验证集数据加载器

                参数:
                    image_label_pairs: 图像和标签路径的元组列表
        """
        self.image_label_pairs = image_label_pairs

    def __getitem__(self, index):
        """获取单个验证数据样本，不进行数据增强"""

        # 读取图像和标签数据
        image_path = self.image_label_pairs[index][0]
        label_path = self.image_label_pairs[index][1]
        image_array = np.float32(np.load(image_path))
        label_array = np.float32(np.load(label_path))
        # 添加通道维度并转换为PyTorch张量
        image_data = np.expand_dims(image_array, axis=0)
        label_data = np.expand_dims(label_array, axis=0)
        return torch.from_numpy(image_data), torch.from_numpy(label_data)
    def __len__(self):
        return len(self.image_label_pairs)
