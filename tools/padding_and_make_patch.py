import os
import numpy as np
import SimpleITK as sitk

########################################################################################################################
'''
preprocess step 3
'''
# 医学图像预处理步骤3：分块提取与重叠处理(重叠分块)
# 目的：将大尺寸3D图像分割为固定大小的子块（如96×96×64），便于深度学习模型处理
# 关键点：通过补零（Padding）和重叠采样避免边缘信息丢失或卷积失真
########################################################################################################################

# 定义目标子块尺寸（X, Y, Z方向的体素数）
b_nx, b_ny, b_nz = 128, 128, 16
# 重叠比例：子块之间的重叠区域占比（例如0.5表示重叠50%）
# 重叠采样：相邻子块之间重叠 50%（如步长为子块尺寸的 50%），确保边缘区域的特征在多个子块中被重复采样，避免因单一子块边缘失真导致的信息丢失。
cover_ratio = 0.5
# 计算补零宽度：子块尺寸的1/4（例如96→24，用于边缘补零后保留中心可信区域）
pad_nx, pad_ny, pad_nz = b_nx // 4, b_ny // 4, b_nz // 4 ## //表示整除
########################################################################################################################

# 输入文件夹路径（归一化后的图像和重采样后的标签）
image_folder = "D:\\PythonProject\\MMS\\validation_images_normalized\\"
label_folder = "D:\\PythonProject\\MMS\\validation_masks_resized\\"
image_patch_save_folder = "D:\\PythonProject\\MMS\\validation_images_patches\\"
label_patch_save_folder = "D:\\PythonProject\\MMS\\validation_masks_patches\\"
# 获取文件名列表并排序, 确保图像和标签对应
image_names = sorted(os.listdir(image_folder))
label_names = sorted(os.listdir(label_folder))
########################################################################################################################

# 遍历所有图像-标签对
for idx in range(len(image_names)):
    print("********************************************************")
    print("Now is processing: ", image_names[idx])
    # 读取归一化后的图像和重采样后的标签
    image_path = image_folder + image_names[idx]
    label_path = label_folder + label_names[idx]
    image_sitk = sitk.ReadImage(image_path)
    label_sitk = sitk.ReadImage(label_path)
    # 将SimpleITK图像转换为numpy数组
    image_array = sitk.GetArrayFromImage(image_sitk)
    label_array = sitk.GetArrayFromImage(label_sitk)
    ####################################################################################################################
    ## 1. 边缘补零（Padding）
    # 为什么补零？
    # - 原因1：卷积神经网络（如3D U-Net）在处理边缘时，由于Padding操作会引入人工边界值（如0），导致边缘区域的特征可信度较低。
    # - 原因2：分块时若原始图像尺寸无法被子块尺寸整除，补零可使图像尺寸适配分块计算。。
    # - 操作：在X/Y/Z三个方向的两侧均补零，补零宽度为子块尺寸的1/4（pad_nx=24, pad_ny=24, pad_nz=16）
    image_array = np.pad(image_array, ((pad_nz, pad_nz), (pad_ny, pad_ny), (pad_nx, pad_nx)), mode="constant", constant_values=0)
    label_array = np.pad(label_array, ((pad_nz, pad_nz), (pad_ny, pad_ny), (pad_nx, pad_nx)), mode="constant", constant_values=0)
    # 获取补零后的图像尺寸
    v_nx = image_array.shape[2] # X方向尺寸（列）
    v_ny = image_array.shape[1] # Y方向尺寸（行）
    v_nz = image_array.shape[0] # Z方向尺寸（层）
    ## 2. 计算子块滑动步长（Stride）
    # 核心逻辑：通过重叠比例计算有效滑动步长，确保相邻子块间有重叠区域
    # 公式：步长 = 子块尺寸 × (1 - 重叠比例)
    st_nx = np.int32(np.round(b_nx - cover_ratio * b_nx))
    st_ny = np.int32(np.round(b_ny - cover_ratio * b_ny))
    st_nz = np.int32(np.round(b_nz - cover_ratio * b_nz))
    ## 3. 计算子块数量
    # 公式：子块数量 = floor((图像尺寸 - 子块尺寸) / 步长) + 1
    blks_nx = np.int32(np.floor((v_nx - b_nx) / st_nx) + 1)
    blks_ny = np.int32(np.floor((v_ny - b_ny) / st_ny) + 1)
    blks_nz = np.int32(np.floor((v_nz - b_nz) / st_nz) + 1)

    print("X direction block number: ", blks_nx)
    print("Y direction block number: ", blks_ny)
    print("Z direction block number: ", blks_nz)

    current_patch_id = 0
    # 三维空间遍历所有可能的子块位置
    for z_idx in np.arange(0, blks_nz):
        # 计算当前子块在Z方向的起始位置，防止越界 最后一个块顶着边界往回取一个块长
        z_start = np.min((z_idx * st_nz, v_nz - b_nz))
        for y_idx in np.arange(0, blks_ny):
            # 计算当前子块在Y方向的起始位置，防止越界
            y_start = np.min((y_idx * st_ny, v_ny - b_ny))
            for x_idx in np.arange(0, blks_nx):
                # 计算当前子块在X方向的起始位置，防止越界
                x_start = np.min((x_idx * st_nx, v_nx - b_nx))
                # 从补零后的图像中提取子块
                image_patch = image_array[z_start: z_start + b_nz, y_start: y_start + b_ny, x_start: x_start + b_nx]
                # 从补零后的标签中提取对应子块
                label_patch = label_array[z_start: z_start + b_nz, y_start: y_start + b_ny, x_start: x_start + b_nx]

                # 以numpy格式保存图像和标签子块
                np.save(image_patch_save_folder + "mms_train_" + str(idx).rjust(4, '0') + "_" + str(current_patch_id).rjust(4, '0') + '.npy', image_patch)
                np.save(label_patch_save_folder + "mms_train_" + str(idx).rjust(4, '0') + "_" + str(current_patch_id).rjust(4, '0') + '.npy', label_patch)
                current_patch_id = current_patch_id + 1

