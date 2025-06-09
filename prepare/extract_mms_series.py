import os
import numpy as np
import SimpleITK as sitk

########################################################################################################################
'''
因为MMS是序列数据，标注可能包含一个或多个序列，并不固定
preprocess step 2
'''
########################################################################################################################

# 定义数据路径
image_folder = "D:\\zhuomian\\MMS\\testing_images\\"  # 原始图像文件夹路径
mask_folder = "D:\\zhuomian\\MMS\\testing_masks\\"  # 原始掩码文件夹路径
image_save_folder = "D:\\zhuomian\\MMS\\testing_images_series\\"  # 处理后图像保存路径
mask_save_folder = "D:\\zhuomian\\MMS\\testing_masks_series\\"  # 处理后掩码保存路径

# 获取文件夹中的所有文件并排序
image_names = sorted(os.listdir(image_folder))  # 按文件名排序的图像文件列表
mask_names = sorted(os.listdir(mask_folder))  # 按文件名排序的掩码文件列表

# 遍历所有图像-掩码对
for idx in range(len(image_names)):
    print("*******************************************************")
    print("Now is processing: ", image_names[idx])
    image_path = image_folder + image_names[idx]

    # 尝试读取图像，若读取失败则跳过当前文件
    try:
        image_sitk = sitk.ReadImage(image_path)  # 读取医学图像文件
    except RuntimeError:
        continue

    image_array = sitk.GetArrayFromImage(image_sitk)  # 将SimpleITK图像转换为NumPy数组

    # 读取对应的掩码文件
    mask_path = mask_folder + mask_names[idx]
    mask_sitk = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask_sitk)

    ####################################################################################################################
    # 处理每个序列中的切片
    for series_idx in range(image_array.shape[0]):
        mask_series_idx = mask_array[series_idx]  # 获取当前切片的掩码

        # 检查掩码中是否包含4个不同的标签值(0背景+3个器官)，不满足则跳过
        if len(np.unique(mask_series_idx)) != 4:
            continue

        image_series_idx = image_array[series_idx]  # 获取当前切片的图像

        ################################################################################################################
        # 将NumPy数组转换回SimpleITK图像格式
        image_series_sitk = sitk.GetImageFromArray(image_series_idx)
        mask_series_sitk = sitk.GetImageFromArray(mask_series_idx)

        # 设置图像的空间间距信息(从原始图像继承)
        image_series_sitk.SetSpacing(image_sitk.GetSpacing()[:3])
        mask_series_sitk.SetSpacing(image_sitk.GetSpacing()[:3])

        # 保存处理后的图像和掩码
        image_save_path = image_save_folder + image_names[idx][:-7] + "_" + str(series_idx) + ".nii.gz"
        mask_save_path = mask_save_folder + mask_names[idx][:-7] + "_" + str(series_idx) + ".nii.gz"
        sitk.WriteImage(image_series_sitk, image_save_path)
        sitk.WriteImage(mask_series_sitk, mask_save_path)
