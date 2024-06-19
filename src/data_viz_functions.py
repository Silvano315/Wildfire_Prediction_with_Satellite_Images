import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
import os
import pandas as pd
import numpy as np


# Plot randomized images for each folder and for each label
def plot_sample_images(data_dir, splits, categories):

    fig, axes = plt.subplots(len(splits), len(categories), figsize=(10, 5 * len(splits)))

    for i, split in enumerate(splits):
        for j, category in enumerate(categories):
            path = os.path.join(data_dir, split, category)
            sample_image = cv2.imread(os.path.join(path, random.choice(os.listdir(path))))
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
            axes[i, j].imshow(sample_image)
            axes[i, j].set_title(f"{split.capitalize()} - {category.capitalize()}")
            axes[i, j].axis('off')
    plt.show()


# Plot class distribution
def plot_class_distribution(data_dir, splits, categories):

    image_counts = {}

    for split in splits:
        for category in categories:
            key = f"{split}_{category}"
            path = os.path.join(data_dir, split, category)
            image_counts[key] = len(os.listdir(path))

    data = []
    for split in splits:
        for category in categories:
            count = image_counts[f"{split}_{category}"]
            data.append({'Split': split, 'Category': category, 'Count': count})

    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Split', y='Count', hue='Category', data=df, palette='pastel')
    plt.title('Class Distribution in Train, Test, and Validation Sets')
    plt.show()


# Bar plot to see distribution for brightness and contrast
def compute_brightness_contrast(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    brightness = image.mean()
    contrast = image.std()
    return brightness, contrast

# Identify edges using Canny's filter and density
def compute_edge_density(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(image, 100, 200)
    edge_density = edges.sum() / (image.shape[0] * image.shape[1])
    return edge_density

# Evaluate sharpness images using Laplacian operator
def compute_sharpness(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sharpness = cv2.Laplacian(image, cv2.CV_64F).var()
    return sharpness


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


# Function to load a random image 
def load_random_image(images_path):

    splits = ['train', 'test', 'valid']
    categories = ['wildfire', 'nowildfire']

    random_split = random.choice(splits)
    random_category = random.choice(categories)

    image_dir = os.path.join(images_path, random_split, random_category)
    images = os.listdir(image_dir)
    random_image_name = random.choice(images)
    image_path = os.path.join(image_dir, random_image_name)
    
    return cv2.imread(image_path)

# Function to display images (original vs enhanced)
def display_images(original_image, enhanced_image, title):

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    axs[1].imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
    axs[1].set_title(title)
    axs[1].axis('off')
    
    plt.show()