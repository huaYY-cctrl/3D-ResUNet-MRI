import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from collections import OrderedDict


def compute_dice_coefficient(mask_gt, mask_pred):
    """Calculate Dice Coefficient between two binary masks"""
    if mask_gt.shape != mask_pred.shape:
        return
    volume_intersect = (mask_gt & mask_pred).sum()
    volume_sum = mask_gt.sum() + mask_pred.sum()
    return 2 * volume_intersect / volume_sum


# Define list of heart structures (Left Ventricle, Right Ventricle, Myocardium)
# Corresponds to label values (1-3) in segmentation masks
organ_names = ["LV", "RV", "MYO"]

# Initialize ordered dictionary to store evaluation metrics for all samples
seg_metrics = OrderedDict()
seg_metrics['Name'] = list()  # Store sample names
for organ in organ_names:
    seg_metrics['{}_DSC'.format(organ)] = list()  # Add DSC column for each organ

# Set paths to ground truth and predicted segmentation folders
gt_folder = "D:\\PythonProject\\MMS\\test_masks_series\\"
pred_folder = "D:\\PythonProject\\MMS\\testing_images_pred\\"

# Get and sort all files in the folders
gt_names = sorted(os.listdir(gt_folder))
pred_names = sorted(os.listdir(pred_folder))

# Loop through all samples
for idx in range(len(gt_names)):
    # Read ground truth and predicted segmentation images
    gt_sitk = sitk.ReadImage(gt_folder + gt_names[idx])
    gt_data = sitk.GetArrayFromImage(gt_sitk)
    pred_sitk = sitk.ReadImage(pred_folder + pred_names[idx])
    pred_data = sitk.GetArrayFromImage(pred_sitk)

    # Skip if shapes don't match
    if gt_data.shape != pred_data.shape:
        continue

    seg_metrics['Name'].append(gt_names[idx])  # Record current sample name

    # Calculate DSC for each organ
    for i in np.arange(1, 4):
        gt_i = gt_data == i  # Extract binary mask for current organ (ground truth)
        pred_i = pred_data == i  # Extract binary mask for current organ (prediction)

        # Handle special cases:
        if np.sum(gt_i) == 0 and np.sum(pred_i) == 0:
            dsc = 1  # Both are empty, considered perfect match
        elif np.sum(gt_i) == 0 and np.sum(pred_i) > 0:
            dsc = 0  # False positive
        elif np.sum(gt_i) > 0 and np.sum(pred_i) == 0:
            dsc = 0  # False negative
        else:
            dsc = compute_dice_coefficient(gt_i, pred_i)  # Normal Dice calculation

        # Store result in dictionary
        seg_metrics['{}_DSC'.format(organ_names[i - 1])].append(dsc)

# Convert results to DataFrame and save as CSV file
dataframe = pd.DataFrame(seg_metrics)
dataframe.to_csv("dsc.csv", index=False)
