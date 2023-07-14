import cv2
import os

# Set the path to your dataset directory
dataset_dir = r"C:\Users\Acer\Desktop\data_work\2018\test_train_val\balanced2018\Nouveau dossier\test\vasc"

# Set the path to the directory where you want to save the preprocessed images
#output_dir = r"C:\Users\Acer\Desktop\Nouveau dossier (2)\N_G\N_G_valid\vasc"
output_dir1 = r"C:\Users\Acer\Desktop\Nouveau dossier (2)\LAB\LAB_valid\vasc"
output_dir2= r"C:\Users\Acer\Desktop\data_work\2018\test_train_val\balanced2018\median\test\vasc"



# Iterate over the images in the dataset directory
for image_file in os.listdir(dataset_dir):
    # Read the image
    image_path = os.path.join(dataset_dir, image_file)
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Apply Gaussian blur with a 5x5 kernel
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply median blur with a kernel size of 3
    median_blurred_image = cv2.medianBlur(image, 3)

    # Save the preprocessed images to the output directory
    #output_filename = os.path.join(output_dir, image_file)
    #cv2.imwrite(output_filename, blurred_image) # Change this line to save the desired preprocessed image

    # Optionally, save the other preprocessed images as well
    #cv2.imwrite(os.path.join(output_dir, "gray_" + image_file), gray_image)
    cv2.imwrite(os.path.join(output_dir1, "lab_" + image_file), lab_image)
    cv2.imwrite(os.path.join(output_dir2, "median_blur_" + image_file), median_blurred_image)


