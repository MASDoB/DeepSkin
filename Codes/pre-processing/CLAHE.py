import os
import cv2

# Path to your dataset folder
dataset_path = r'C:\Users\Acer\Desktop\data_work\2018\test_train_val\isic2018\akiec'

# List all image file names in the dataset folder
image_files = os.listdir(dataset_path)

# Parameters for adaptive histogram equalization
#can adjust the clip_limit and tile_grid_size parameters to control the level of enhancement.
clip_limit = 2.0
tile_grid_size = (8,8)

# Load and process each image in the dataset
enhanced_images = []
for image_file in image_files:
    image_path = os.path.join(dataset_path, image_file)
    image = cv2.imread(image_path, 0)  # Load image in grayscale

    # Apply adaptive histogram equalization
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_image = clahe.apply(image)

    enhanced_images.append(enhanced_image)

# Save the enhanced images
output_path = r'C:\Users\Acer\Desktop\Nouveau dossier (4)\akiec stat'
os.makedirs(output_path, exist_ok=True)

for i, enhanced_image in enumerate(enhanced_images):
    output_file = os.path.join(output_path, f'enhanced_image_{i}.jpg')
    cv2.imwrite(output_file, enhanced_image)