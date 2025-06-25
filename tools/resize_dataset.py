import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom


########################################################################################################################
'''
preprocess step 1
'''
# 医学图像预处理步骤1：将所有图像和标签重采样(缩放)到统一的体素尺寸
# 这是医学图像处理中的关键步骤，可确保不同来源的图像具有相同的空间分辨率
########################################################################################################################

########################################################################################################################
# 设置目标体素尺寸（毫米），X、Y、Z方向 ，对应值是通过Excel数据集体素尺寸求平均值得来
# 这是预处理的目标分辨率，所有图像都将被重采样到这个体素尺寸
new_voxel = [1.3056, 1.3056, 9.5422]
########################################################################################################################
# 原始图像和标签所在的文件夹路径
ori_image_folder = "D:\\PythonProject\\MMS\\validation_images_series\\"
ori_label_folder = "D:\\PythonProject\\MMS\\validation_masks_series\\"
# 获取文件夹中所有文件的名称并排序，确保图像和标签对应
image_names = sorted(os.listdir(ori_image_folder))
label_names = sorted(os.listdir(ori_label_folder))
# 重采样(缩放)后图像和标签的保存路径
image_save_folder = "D:\\PythonProject\\MMS\\validation_images_resized\\"
label_save_folder = "D:\\PythonProject\\MMS\\validation_masks_resized\\"
########################################################################################################################

# 遍历所有图像和标签文件
for idx in range(len(image_names)):
    print("********************************************************")
    print("Now is processing: ", image_names[idx])
    # 读取原始图像和标签
    image_path = ori_image_folder + image_names[idx]
    image_sitk = sitk.ReadImage(image_path) # 使用SimpleITK读取图像
    image_array = sitk.GetArrayFromImage(image_sitk) # 转换为numpy数组 (Z,Y,X)

    label_path = ori_label_folder + label_names[idx]
    label_sitk = sitk.ReadImage(label_path) # 使用SimpleITK读取标签
    label_array = sitk.GetArrayFromImage(label_sitk) # 转换为numpy数组 (Z,Y,X)
    ####################################################################################################################
    old_voxel = image_sitk.GetSpacing() # 获取原始体素尺寸(X,Y,Z)
    # 计算重采样因子：新体素尺寸与旧体素尺寸的比值（计算依据：体素个数*体素尺寸不变）
    # 注意：SimpleITK的Spacing顺序是(X,Y,Z)，而numpy数组的顺序是(Z,Y,X)
    # 使用scipy.ndimage.zoom进行重采样
    # 图像使用3次样条插值（order=3）以保留更多细节
    image_resized = zoom(input=image_array, zoom=[old_voxel[2] / new_voxel[2], old_voxel[1] / new_voxel[1], old_voxel[0]/ new_voxel[0]], order=3)
    # 标签使用最近邻插值（order=0）以避免引入新的标签值
    label_resized = zoom(input=label_array, zoom=[old_voxel[2] / new_voxel[2], old_voxel[1] / new_voxel[1], old_voxel[0] / new_voxel[0]], order=0)
    image_resized = np.float32(image_resized) # 图像使用32位浮点数
    label_resized = np.uint8(label_resized) # 标签使用8位无符号整数
    ####################################################################################################################
    # 将numpy数组转换回SimpleITK图像对象
    image_resized_sitk = sitk.GetImageFromArray(image_resized)
    label_resized_sitk = sitk.GetImageFromArray(label_resized)
    # 设置新的体素尺寸信息
    image_resized_sitk.SetSpacing(new_voxel)
    label_resized_sitk.SetSpacing(new_voxel)
    # 保存重采样后的图像和标签
    sitk.WriteImage(image=image_resized_sitk, fileName=image_save_folder + image_names[idx])
    sitk.WriteImage(image=label_resized_sitk, fileName=label_save_folder + label_names[idx])
