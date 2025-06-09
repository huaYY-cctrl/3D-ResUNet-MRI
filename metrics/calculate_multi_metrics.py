import os
import gc
import numpy as np
import pandas as pd
import SimpleITK as sitk
from collections import OrderedDict


def compute_dice_coefficient(mask_gt, mask_pred):
    if mask_gt.shape != mask_pred.shape:
        return
    """计算两个二值掩码之间的Dice系数"""
    volume_intersect = (mask_gt & mask_pred).sum()
    volume_sum = mask_gt.sum() + mask_pred.sum()
    return 2 * volume_intersect / volume_sum


# 定义左心室、右心室、心肌列表，与标签编号(1-3)对应
organ_names = ["LV", "RV", "MYO"]

# 初始化有序字典，用于存储所有样本的评估指标
seg_metrics = OrderedDict()
seg_metrics['Name'] = list()  # 存储样本名称
for organ in organ_names:
    seg_metrics['{}_DSC'.format(organ)] = list()  # 为每个器官添加DSC指标列

# 设置金标准和预测结果的文件夹路径
gt_folder = "D:\\zhuomian\\MMS\\testing_masks_series\\"
pred_folder = "D:\\zhuomian\\MMS\\testing_images_pred\\"

# 获取并排序文件夹中的所有文件
gt_names = sorted(os.listdir(gt_folder))
pred_names = sorted(os.listdir(pred_folder))

# 遍历所有样本
for idx in range(len(gt_names)):
    # 读取金标准和预测分割图像
    gt_sitk = sitk.ReadImage(gt_folder + gt_names[idx])
    gt_data = sitk.GetArrayFromImage(gt_sitk)
    pred_sitk = sitk.ReadImage(pred_folder + pred_names[idx])
    pred_data = sitk.GetArrayFromImage(pred_sitk)
    if gt_data.shape != pred_data.shape:
        continue
    seg_metrics['Name'].append(gt_names[idx])  # 记录当前样本名称

    # 计算每个器官的DSC指标
    for i in np.arange(1, 4):
        gt_i = gt_data == i  # 提取当前器官的金标准二值掩码
        pred_i = pred_data == i  # 提取当前器官的预测二值掩码

        # 处理特殊情况：
        if np.sum(gt_i) == 0 and np.sum(pred_i) == 0:
            dsc = 1  # 两者都不存在，视为完全匹配
        elif np.sum(gt_i) == 0 and np.sum(pred_i) > 0:
            dsc = 0  # 假阳性
        elif np.sum(gt_i) > 0 and np.sum(pred_i) == 0:
            dsc = 0  # 假阴性
        else:
            dsc = compute_dice_coefficient(gt_i, pred_i)  # 正常计算Dice系数

        # 将计算结果存入字典
        seg_metrics['{}_DSC'.format(organ_names[i - 1])].append(dsc)

# 将结果转换为DataFrame并保存为CSV文件
dataframe = pd.DataFrame(seg_metrics)
dataframe.to_csv("dsc.csv", index=False)
