import cv2
import os

# Path to the directory containing the images
dataset_path = r'C:\Users\Acer\Desktop\DATAsets\ISIC2018\test_train_val\s_c_test\vasc'
output_dir =r'C:\Users\Acer\Desktop\Nouveau dossier (2)\HSV\HSV_test\vasc'
# Get a list of all image files in the directory
image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg') or f.endswith('.png')]

# Process each image
for image_file in image_files:
    # Load the image
    image_path = os.path.join(dataset_path, image_file)
    image = cv2.imread(image_path)
    
    # Convert the image to HSV color mode
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Save the HSV image
    hsv_image_path = os.path.join(output_dir , 'hsv_' + image_file)
    cv2.imwrite(hsv_image_path, hsv_image)
    


