from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import shutil

# Directory containing the original images
original_images_dir = r"C:\Users\Acer\Desktop\data_work\2018\test_train_val\bi_mal\1"

# Directory to store augmented images
augmented_images_dir = r"C:\Users\Acer\Desktop\data_work\2018\test_train_val\bi_mal\malignant"

number_original_images=2305
number_desired_images=9415
# Number of desired augmented images
desired_augmented_images = number_desired_images-number_original_images

# Create a temporary directory to store original images
temp_original_images_dir = "path/to/temp/original/images15"

# Copy the original images to the temporary directory
shutil.copytree(original_images_dir, temp_original_images_dir)

# Create the data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Create a generator for the original images
original_generator = datagen.flow_from_directory(
    temp_original_images_dir,
    target_size=(450, 600),
    batch_size=1,
    class_mode=None,
    shuffle=False
)

# Ensure the augmented images directory exists
os.makedirs(augmented_images_dir, exist_ok=True)

# Save the original images to the augmented images directory
for i, image_path in enumerate(original_generator.filenames):
    shutil.copy(os.path.join(temp_original_images_dir, image_path), os.path.join(augmented_images_dir, f"original_{i}.jpg"))

# Perform data augmentation until the desired number of augmented images is reached
augmented_images_count = 0
for i in range(desired_augmented_images):
    # Generate an augmented image
    augmented_image_array = next(original_generator)
    
    # Convert the NumPy array to an Image object
    augmented_image = Image.fromarray(augmented_image_array[0].astype('uint8'))
    
    # Save the augmented image to the augmented images directory
    augmented_image_path = os.path.join(augmented_images_dir, f"augmented_{i}.jpg")
    augmented_image.save(augmented_image_path)

    augmented_images_count += 1

    # Break the loop if the desired number of augmented images is reached
    if augmented_images_count == desired_augmented_images:
        break

print("Data augmentation completed.")