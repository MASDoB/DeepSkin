import os
import numpy as np
import cv2

def gamut_mapping(image, target_mean, target_std):
    # Convert the image to float for accurate calculations
    image = image.astype(np.float64)

    # Calculate the mean and standard deviation of the image
    image_mean = np.mean(image)
    image_std = np.std(image)

    # Calculate the normalized image
    normalized_image = (image - image_mean) / image_std

    # Calculate the enhanced image using gamut mapping
    enhanced_image = (target_std * normalized_image) + target_mean

    # Clip the pixel values to ensure they are within the valid range [0, 255]
    enhanced_image = np.clip(enhanced_image, 0, 255)

    # Convert the image data type to uint8
    enhanced_image = enhanced_image.astype(np.uint8)

    return enhanced_image

# Path to your dataset folder
dataset_path = r'C:\Users\Acer\Desktop\data_work\2018\test_train_val\isic2018\akiec'

# List all image file names in the dataset folder
image_files = os.listdir(dataset_path)

# Target mean and standard deviation for gamut mapping
target_mean = 128.0
target_std = 52.0

# Load and process each image in the dataset
enhanced_images = []
for image_file in image_files:
    image_path = os.path.join(dataset_path, image_file)
    image = cv2.imread(image_path)
    enhanced_image = gamut_mapping(image, target_mean, target_std)
    enhanced_images.append(enhanced_image)

# Save the enhanced images
output_path = r'C:\Users\Acer\Desktop\Nouveau dossier (4)\akiec gamur'
os.makedirs(output_path, exist_ok=True)

for i, enhanced_image in enumerate(enhanced_images):
    output_file = os.path.join(output_path, f'enhanced_image_{i}.jpg')
    cv2.imwrite(output_file, enhanced_image)

