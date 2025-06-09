import os
import torch
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
from models.UNet3DMMS import UNet3DMMS


data_folder = r"D:\zhuomian\MMS\testing_images_series" # 原始医学图像路径（NIfTI格式）
pred_folder = r"D:\zhuomian\MMS\testing_images_pred" # 预测结果保存路径
new_voxel = [1.3064, 1.3064, 9.5391]
b_nx, b_ny, b_nz = 128, 128, 16
st_nx, st_ny, st_nz = 64, 64, 8
pad_nx, pad_ny, pad_nz = 32, 32, 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#############################################################################
model = UNet3DMMS(input_ch=1, output_ch=4).to(device)
model.load_state_dict(torch.load(r'D:\zhuomian\PycharmProjects\PythonProject\CardiacStructSeg\modelsave\UNet\MMS\UNet_000281.pth', map_location="cpu"))
model.eval()
#############################################################################
# 读取原始图像
data_names = sorted(os.listdir(data_folder))
for data_name in data_names:
    print(data_name)
    data_path = data_folder + '\\' + data_name
    image_sitk = sitk.ReadImage(data_path) # 使用SimpleITK读取图像
    image_array = sitk.GetArrayFromImage(image_sitk) # 转换为numpy数组 (Z,Y,X)
    old_voxel = image_sitk.GetSpacing() # 获取原始体素尺寸(X,Y,Z)
    image_resized = zoom(input=image_array, zoom=[old_voxel[2] / new_voxel[2], old_voxel[1] / new_voxel[1], old_voxel[0]/ new_voxel[0]], order=3)
    ############################################################################
    # 去除图像中的噪点或异常亮 / 暗区域，保留主要信息
    q5 = np.quantile(image_resized, 0.05)  # 计算图像中第 5% 小的像素值（即有 5% 的像素值比它小）
    q95 = np.quantile(image_resized, 0.95) # 计算图像中第 95% 小的像素值（即有 95% 的像素值比它小）
    image_resized[image_resized < q5] = q5  # 将所有小于q5的像素值强制设为q5
    image_resized[image_resized > q95] = q95 # 将所有大于q95的像素值强制设为q95
    image_normalized = (image_resized - np.mean(image_resized)) / np.std(image_resized)
    image_normalized = np.float32(image_normalized) # 转换数据类型为32位浮点数（深度学习模型常用格式 pytorch）
    ############################################################################
    image_padded = np.pad(image_normalized, [(pad_nz, pad_nz), (pad_ny, pad_ny), (pad_nx, pad_nx)], mode="constant", constant_values=0)
    v_nx = image_padded.shape[2] # X方向尺寸（列）
    v_ny = image_padded.shape[1] # Y方向尺寸（行）
    v_nz = image_padded.shape[0] # Z方向尺寸（层）
    blks_nx = np.int32(np.floor((v_nx - b_nx) / st_nx) + 1)
    blks_ny = np.int32(np.floor((v_ny - b_ny) / st_ny) + 1)
    blks_nz = np.int32(np.floor((v_nz - b_nz) / st_nz) + 1)
    ############################################################################
    label_pred = np.zeros(shape=(4, image_normalized.shape[0], image_normalized.shape[1], image_normalized.shape[2]))
    # 三维空间遍历所有可能的子块位置
    for z_idx in np.arange(0, blks_nz):
        # 计算当前子块在Z方向的起始位置，防止越界 最后一个块顶着边界往回取一个块长
        z_start = np.min((z_idx * st_nz, v_nz - b_nz))
        z_start_pred = np.min((z_idx * b_nz / 2, label_pred.shape[1] - b_nz / 2))
        z_start_pred = int(z_start_pred)
        for y_idx in np.arange(0, blks_ny):
            # 计算当前子块在Y方向的起始位置，防止越界
            y_start = np.min((y_idx * st_ny, v_ny - b_ny))
            y_start_pred = np.min((y_idx * b_ny / 2, label_pred.shape[2] - b_ny / 2))
            y_start_pred = int(y_start_pred)
            for x_idx in np.arange(0, blks_nx):
                # 计算当前子块在X方向的起始位置，防止越界
                x_start = np.min((x_idx * st_nx, v_nx - b_nx))
                x_start_pred = np.min((x_idx * b_nx / 2, label_pred.shape[3] - b_nx / 2))
                x_start_pred = int(x_start_pred)
                # 从补零后的图像中提取子块
                image_patch = image_padded[z_start: z_start + b_nz, y_start: y_start + b_ny, x_start: x_start + b_nx]
                image_patch = np.reshape(image_patch, [1, 1, b_nz, b_ny, b_nx])
                image_patch = torch.from_numpy(image_patch)
                image_patch = image_patch.to(device)
                pred_patch = model(image_patch)
                label_pred[:, z_start_pred: z_start_pred + int(b_nz / 2), y_start_pred: y_start_pred + int(b_ny / 2), x_start_pred: x_start_pred + int(b_nx / 2)] = pred_patch[0][:, 4:12, 32:96, 32:96].cpu().detach().numpy()
    # --------------------------- 结果后处理与保存 ---------------------------
    # 对类别维度取最大值，得到每个体素的预测类别（维度：Z×Y×X）
    label_pred = np.argmax(label_pred, axis=0)
    # 将预测结果尺寸恢复为原始图像的体素尺寸（最近邻插值，保持类别标签离散）
    label_resized = zoom(
        input=label_pred,
        zoom=[new_voxel[2] / old_voxel[2], new_voxel[1] / old_voxel[1], new_voxel[0]/ old_voxel[0]],
        order=0  # 最近邻插值（适用于离散标签）
    )
    # 转换为uint8类型（符合医学图像标签存储格式）
    label_resized = np.uint8(label_resized)

    # 将预测结果转换为SimpleITK图像对象并设置元数据
    label_resized_sitk = sitk.GetImageFromArray(label_resized)  # 数组转图像（维度顺序Z,Y,X）
    label_resized_sitk.SetOrigin(image_sitk.GetOrigin())          # 设置原点坐标
    label_resized_sitk.SetSpacing(image_sitk.GetSpacing())        # 设置体素尺寸（恢复原始尺寸）
    label_resized_sitk.SetDirection(image_sitk.GetDirection())    # 设置方向矩阵

    # 保存预测结果为NIfTI格式文件
    sitk.WriteImage(label_resized_sitk, pred_folder + "\\" + data_name)
