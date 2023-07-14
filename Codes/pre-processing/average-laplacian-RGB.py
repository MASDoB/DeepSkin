import cv2
import os

# Function to apply Laplacian filter to an image
def apply_laplacian_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_image = cv2.Laplacian(gray, cv2.CV_64F)
    filtered_image = cv2.normalize(filtered_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return filtered_image

# Function to apply average filter to an image
def apply_average_filter(image, kernel_size):
    filtered_image = cv2.blur(image, (kernel_size, kernel_size))
    return filtered_image

# Set the path to the directory containing the image dataset
dataset_dir = r'C:\Users\Acer\Desktop\data_work\70_15_15\test\akiec'  # Replace with the path to your image dataset directory

# Set the kernel size for the average filter
kernel_size = 5

# Set the path to the directory to save the subtracted images
output_dir = r'C:\Users\Acer\Desktop\2019.v2\Gabor_filter\test\AK'  # Replace with the path to your output directory

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over the images in the dataset directory
for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpg'):  # Adjust the file extension if necessary
        # Read the image
        image_path = os.path.join(dataset_dir, filename)
        image = cv2.imread(image_path)

        # Apply the average filter to the image
        avg_filtered_image = apply_average_filter(image, kernel_size)

        # Apply the Laplacian filter to the image
        lap_filtered_image = apply_laplacian_filter(image)

        # Convert the Laplacian filtered image to 3 channels (RGB)
        lap_filtered_image = cv2.cvtColor(lap_filtered_image, cv2.COLOR_GRAY2BGR)

        # Subtract the Laplacian filtered image from the average filtered image
        subtracted_image = cv2.subtract(avg_filtered_image, lap_filtered_image)

        # Save the subtracted image in the output directory
        subtracted_image_path = os.path.join(output_dir, 'subtracted_' + filename)
        cv2.imwrite(subtracted_image_path, subtracted_image)
