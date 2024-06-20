import cv2
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import random
from matplotlib import pyplot as plt


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