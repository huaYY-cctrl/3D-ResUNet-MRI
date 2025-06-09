import os
import numpy as np
import SimpleITK as sitk


########################################################################################################################
'''
preprocess step 2
'''
# 医学图像预处理步骤2：标准化（归一化）处理
# 对重采样后的图像进行强度归一化，确保深度学习模型输入数据的一致性
########################################################################################################################

# 原始图像文件夹路径（重采样后的图像）
ori_image_folder = "D:\\zhuomian\\MMS\\validation_images_resized\\"
# 归一化后图像的保存路径
image_save_folder = "D:\\zhuomian\\MMS\\validation_images_normalized\\"
# 获取文件夹中所有图像文件名并排序
image_names = sorted(os.listdir(ori_image_folder))
# 遍历处理所有图像
for idx in range(len(image_names)):
    print("********************************************************")
    print("Now is processing: ", image_names[idx])
    # 读取重采样后的图像（已调整体素尺寸）
    image_path = ori_image_folder + image_names[idx]
    image_sitk = sitk.ReadImage(image_path) # 使用SimpleITK读取图像
    image_array = sitk.GetArrayFromImage(image_sitk) # 转换为NumPy数组（Z, Y, X顺序）
    ## 第一步：调窗处理
    # 作用：将图像强度值截断到特定范围，突出感兴趣组织
    q5 = np.quantile(image_array, 0.05)
    q95 = np.quantile(image_array, 0.95)
    image_array[image_array < q5] = q5
    image_array[image_array > q95] = q95
    image_array = (image_array - np.mean(image_array)) / np.std(image_array)
    image_norm_sitk = sitk.GetImageFromArray(image_array) # 将NumPy数组转换回SimpleITK图像对象
    # 复制原始图像的空间信息（原点、体素尺寸、方向），确保空间定位不变
    image_norm_sitk.SetOrigin(image_sitk.GetOrigin())
    image_norm_sitk.SetSpacing(image_sitk.GetSpacing())
    image_norm_sitk.SetDirection(image_sitk.GetDirection())
    # 保存归一化后的图像
    sitk.WriteImage(image=image_norm_sitk, fileName=image_save_folder + image_names[idx])
