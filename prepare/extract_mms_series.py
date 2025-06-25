import os
import numpy as np
import SimpleITK as sitk

########################################################################################################################
'''
因为MMS是序列数据，标注可能包含一个或多个序列，并不固定
Preprocess step 2
'''
########################################################################################################################

# Define data paths
image_folder = "D:\\PythonProject\\MMS\\test_images\\"  # Path to the original image folder
mask_folder = "D:\\PythonProject\\MMS\\test_masks\\"  # Path to the original mask folder
image_save_folder = "D:\\PythonProject\\MMS\\test_images_series\\"  # Path to save processed images
mask_save_folder = "D:\\PythonProject\\MMS\\test_masks_series\\"  # Path to save processed masks

# Get all files in the folder and sort them
image_names = sorted(os.listdir(image_folder))  # List of image files sorted by filename
mask_names = sorted(os.listdir(mask_folder))  # List of mask files sorted by filename

# Iterate through all image-mask pairs
for idx in range(len(image_names)):
    print("*******************************************************")
    print("Now is processing: ", image_names[idx])
    image_path = image_folder + image_names[idx]

    # Try to read the image, skip the current file if reading fails
    try:
        image_sitk = sitk.ReadImage(image_path)  # Read medical image file
    except RuntimeError:
        continue

    image_array = sitk.GetArrayFromImage(image_sitk)  # Convert SimpleITK image to NumPy array

    # Read the corresponding mask file
    mask_path = mask_folder + mask_names[idx]
    mask_sitk = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask_sitk)

    ####################################################################################################################
    # Process each slice in the sequence
    for series_idx in range(image_array.shape[0]):
        mask_series_idx = mask_array[series_idx]  # Get the mask for the current slice

        # Check if the mask contains 4 different label values (0 background + 3 organs), skip if not
        if len(np.unique(mask_series_idx)) != 4:
            continue

        image_series_idx = image_array[series_idx]  # Get the image for the current slice

        ################################################################################################################
        # Convert NumPy array back to SimpleITK image format
        image_series_sitk = sitk.GetImageFromArray(image_series_idx)
        mask_series_sitk = sitk.GetImageFromArray(mask_series_idx)

        # Set the spatial spacing information of the image (inherited from the original image)
        image_series_sitk.SetSpacing(image_sitk.GetSpacing()[:3])
        mask_series_sitk.SetSpacing(image_sitk.GetSpacing()[:3])

        # Save the processed image and mask
        image_save_path = image_save_folder + image_names[idx][:-7] + "_" + str(series_idx) + ".nii.gz"
        mask_save_path = mask_save_folder + mask_names[idx][:-7] + "_" + str(series_idx) + ".nii.gz"
        sitk.WriteImage(image_series_sitk, image_save_path)
        sitk.WriteImage(mask_series_sitk, mask_save_path)
