import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from matplotlib import pyplot as plt

from src.constants import BATCH_SIZE, IMAGE_SIZE


# Verify dimensions for every image in the folders
def check_image_dimensions(data_dir, splits, labels, corrupted_files):

    for split in splits:
        for label in labels:
            path = os.path.join(data_dir, split, label)
            for img_name in os.listdir(path):  
                img_path = os.path.join(path, img_name)
                with open(img_path, 'rb') as f:
                     check_chars = f.read()[-2:]
                if check_chars != b'\xff\xd9':
                    print(f"Warning: Unable to read image {img_name} in {split}/{label}. It may be corrupted.")
                    corrupted_files["files"].append(img_name)
                    corrupted_files["paths"].append(f"{split}/{label}")
                else:
                    img = cv2.imread(img_path)
                    if img.shape[:2] != (350, 350):
                        print(f"Image {img_name} in {split}/{label} has dimensions {img.shape[:2]} instead of 350x350.")


# Define the sharpen function
def sharpen(image):

    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image


# Define the noise reduction function
def noise_reduction(image, kernel_size=(5, 5)):

    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    
    return blurred_image


# Data Augmentation - Visualization phase

# Function to plot original and augmented images
def load_and_plot_augmented_images(data_dir, datagen, num_augmented=5):

    splits = ['train']
    labels = ['wildfire', 'nowildfire']
    random_split = random.choice(splits)
    random_label = random.choice(labels)
    label_path = os.path.join(data_dir, random_split, random_label)
    
    images = os.listdir(label_path)
    
    random_image_name = random.choice(images)
    image_path = os.path.join(label_path, random_image_name)
    
    image = cv2.imread(image_path)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, 0) 
    
    fig, axes = plt.subplots(1, num_augmented + 1, figsize=(20, 5))
    axes[0].imshow(image[0])
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    i = 1
    for batch in datagen.flow(image, batch_size=1):
        aug_image = batch[0].astype('uint8')
        axes[i].imshow(aug_image)
        axes[i].set_title('Augmented Image')
        axes[i].axis('off')
        i += 1
        if i > num_augmented:
            break
    plt.show()


# Function to do image pre-processing: data augmentation and normalization
def create_generator(images_path, dataset_type, target_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, num_output=1):

    if dataset_type not in ['train', 'valid', 'test']:
        raise ValueError("Check dataset name: dataset_type must be one of 'train', 'valid', or 'test'")

    dataset_dir = os.path.join(images_path, dataset_type)
    
    if dataset_type == 'train':
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
    else:
        datagen = ImageDataGenerator(rescale=1./255)
    
    generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary' if num_output == 1 else 'categorical'
    )
    
    return generator
