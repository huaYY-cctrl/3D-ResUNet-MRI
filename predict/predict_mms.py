import os
import torch
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
from models.UNet3DMMS import UNet3DMMS

data_folder = "D:\\PythonProject\\MMS\\test_images_series\\"  # Path to original medical images (NIfTI format)
pred_folder = "D:\\PythonProject\\MMS\\testing_images_pred\\"  # Path to save prediction results
new_voxel = [1.3056, 1.3056, 9.5422]
b_nx, b_ny, b_nz = 128, 128, 16
st_nx, st_ny, st_nz = 64, 64, 8
pad_nx, pad_ny, pad_nz = 32, 32, 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
model = UNet3DMMS(input_ch=1, output_ch=4).to(device)
model.load_state_dict(
    torch.load("D:\\PythonProject\\CardiacStructSeg\\modelsave\\UNet\\MMS\\UNet_000281.pth",
               map_location="cuda:0"))
model.eval()

# Read original images
data_names = sorted(os.listdir(data_folder))
for data_name in data_names:
    print(data_name)
    data_path = data_folder + '\\' + data_name
    image_sitk = sitk.ReadImage(data_path)  # Read image using SimpleITK
    image_array = sitk.GetArrayFromImage(image_sitk)  # Convert to numpy array (Z,Y,X)
    old_voxel = image_sitk.GetSpacing()  # Get original voxel size (X,Y,Z)

    # Resize image to new voxel size
    image_resized = zoom(
        input=image_array,
        zoom=[old_voxel[2] / new_voxel[2], old_voxel[1] / new_voxel[1], old_voxel[0] / new_voxel[0]],
        order=3
    )

    # Remove noise or abnormal bright/dark regions, retain main information
    q5 = np.quantile(image_resized, 0.05)  # Calculate the 5th percentile pixel value
    q95 = np.quantile(image_resized, 0.95)  # Calculate the 95th percentile pixel value
    image_resized[image_resized < q5] = q5  # Set all pixels smaller than q5 to q5
    image_resized[image_resized > q95] = q95  # Set all pixels larger than q95 to q95

    # Normalize the image
    image_normalized = (image_resized - np.mean(image_resized)) / np.std(image_resized)
    image_normalized = np.float32(image_normalized)  # Convert to 32-bit floating point (common format for deep learning models in PyTorch)

    # Pad the image to handle boundary conditions
    image_padded = np.pad(
        image_normalized,
        ((pad_nz, pad_nz), (pad_ny, pad_ny), (pad_nx, pad_nx)),
        mode="constant",
        constant_values=0
    )

    # Get dimensions of the padded image
    v_nx = image_padded.shape[2]  # X dimension (columns)
    v_ny = image_padded.shape[1]  # Y dimension (rows)
    v_nz = image_padded.shape[0]  # Z dimension (slices)

    # Calculate the number of blocks in each dimension
    blks_nx = np.int32(np.floor((v_nx - b_nx) / st_nx) + 1)
    blks_ny = np.int32(np.floor((v_ny - b_ny) / st_ny) + 1)
    blks_nz = np.int32(np.floor((v_nz - b_nz) / st_nz) + 1)

    # Initialize prediction array
    label_pred = np.zeros(shape=(4, image_normalized.shape[0], image_normalized.shape[1], image_normalized.shape[2]))

    # Traverse all possible block positions in 3D space
    for z_idx in np.arange(0, blks_nz):
        # Calculate the starting position in Z dimension, prevent out-of-bounds
        z_start = np.min((z_idx * st_nz, v_nz - b_nz))
        z_start_pred = np.min((z_idx * b_nz / 2, label_pred.shape[1] - b_nz / 2))
        z_start_pred = int(z_start_pred)

        for y_idx in np.arange(0, blks_ny):
            # Calculate the starting position in Y dimension, prevent out-of-bounds
            y_start = np.min((y_idx * st_ny, v_ny - b_ny))
            y_start_pred = np.min((y_idx * b_ny / 2, label_pred.shape[2] - b_ny / 2))
            y_start_pred = int(y_start_pred)

            for x_idx in np.arange(0, blks_nx):
                # Calculate the starting position in X dimension, prevent out-of-bounds
                x_start = np.min((x_idx * st_nx, v_nx - b_nx))
                x_start_pred = np.min((x_idx * b_nx / 2, label_pred.shape[3] - b_nx / 2))
                x_start_pred = int(x_start_pred)

                # Extract patch from padded image
                image_patch = image_padded[z_start: z_start + b_nz, y_start: y_start + b_ny, x_start: x_start + b_nx]
                image_patch = np.reshape(image_patch, [1, 1, b_nz, b_ny, b_nx])
                image_patch = torch.from_numpy(image_patch)
                image_patch = image_patch.to(device)

                # Get model prediction
                pred_patch = model(image_patch)

                # Update prediction array
                label_pred[:, z_start_pred: z_start_pred + int(b_nz / 2), y_start_pred: y_start_pred + int(b_ny / 2),
                x_start_pred: x_start_pred + int(b_nx / 2)] = pred_patch[0][:, 4:12, 32:96,
                                                              32:96].cpu().detach().numpy()

    # --------------------------- Post-processing and saving results ---------------------------
    # Get the predicted class for each voxel by taking the argmax along the class dimension (shape: Z×Y×X)
    label_pred = np.argmax(label_pred, axis=0)

    # Resize the prediction back to the original voxel size (nearest neighbor interpolation to preserve discrete class labels)
    label_resized = zoom(
        input=label_pred,
        zoom=[new_voxel[2] / old_voxel[2], new_voxel[1] / old_voxel[1], new_voxel[0] / old_voxel[0]],
        order=0  # Nearest neighbor interpolation (suitable for discrete labels)
    )

    # Convert to uint8 type (suitable for medical image label storage)
    label_resized = np.uint8(label_resized)

    # Convert the prediction to a SimpleITK image and set metadata
    label_resized_sitk = sitk.GetImageFromArray(label_resized)  # Convert array to image (dimension order Z,Y,X)
    label_resized_sitk.SetOrigin(image_sitk.GetOrigin())  # Set origin coordinates
    label_resized_sitk.SetSpacing(image_sitk.GetSpacing())  # Set voxel size (restore original size)
    label_resized_sitk.SetDirection(image_sitk.GetDirection())  # Set direction matrix

    # Save the prediction as a NIfTI file
    sitk.WriteImage(label_resized_sitk, pred_folder + "\\" + data_name)
