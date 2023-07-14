import os
import cv2

# Path to your dataset folder
dataset_path = r'C:\Users\Acer\Desktop\data_work\2018\test_train_val\isic2018\akiec'

# List all image file names in the dataset folder
image_files = os.listdir(dataset_path)

# Parameters for CLAHE
clip_limit = 2.0
tile_grid_size = (8, 8)

# Parameters for Unsharp Masking
kernel_size = (5, 5)
sigma = 1.0
amount = 1.5
threshold = 0

# Load and process each image in the dataset
enhanced_images = []
for image_file in image_files:
    image_path = os.path.join(dataset_path, image_file)
    image = cv2.imread(image_path)

    # Convert image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Apply CLAHE on the L channel
    #can adjust the parameters for CLAHE and Unsharp Masking to achieve the desired enhancement level.
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:,:,0] = clahe.apply(lab[:,:,0])

    # Convert image back to BGR color space
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Apply Unsharp Masking for enhanced details
    blurred = cv2.GaussianBlur(enhanced_image, kernel_size, sigma)
    unsharp_mask = cv2.addWeighted(enhanced_image, 1 + amount, blurred, -amount, threshold)
    enhanced_images.append(unsharp_mask)

# Save the enhanced images
output_path = r'C:\Users\Acer\Desktop\Nouveau dossier (4)\akiec hybrid'
os.makedirs(output_path, exist_ok=True)

for i, enhanced_image in enumerate(enhanced_images):
    output_file = os.path.join(output_path, f'enhanced_image_{i}.jpg')
    cv2.imwrite(output_file, enhanced_image)
