import os
import shutil
import numpy as np


########################################################################################################################
'''
preprocess step 1
'''
########################################################################################################################


dataset_folders_folder = "D:\\zhuomian\\MMS\\Testing\\"
dataset_folders = sorted(os.listdir(dataset_folders_folder))
image_save_folder = "D:\\zhuomian\\MMS\\testing_images\\"
mask_save_folder = "D:\\zhuomian\\MMS\\testing_masks\\"

for dataset_folder in dataset_folders:
    dataset_folder_all = dataset_folders_folder + dataset_folder + "\\"
    dataset_file_names = sorted(os.listdir(dataset_folder_all))
    for dataset_file_name in dataset_file_names:
        if "_gt.nii.gz" in dataset_file_name:
            shutil.copy(dataset_folder_all + dataset_file_name, mask_save_folder + dataset_file_name)
        else:
            shutil.copy(dataset_folder_all + dataset_file_name, image_save_folder + dataset_file_name)
