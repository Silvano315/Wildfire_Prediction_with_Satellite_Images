import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


# Function to load a random image 
def load_random_image(images_path):

    splits = ['train', 'test', 'valid']
    labels = ['wildfire', 'nowildfire']

    random_split = random.choice(splits)
    random_label = random.choice(labels)

    image_dir = os.path.join(images_path, random_split, random_label)
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

def compute_mean_color_histogram(images_path, split_name, category_name):

    dir_path = os.path.join(images_path, split_name, category_name)
    
    mean_hist_b = np.zeros((256,))
    mean_hist_g = np.zeros((256,))
    mean_hist_r = np.zeros((256,))
    
    num_images = 0
    
    image_files = os.listdir(dir_path)
    for image_name in image_files:
        image_path = os.path.join(dir_path, image_name)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Warning: Could not read image '{image_path}'. Skipping...")
            continue
        
        channels = cv2.split(image)
        
        # Compute histograms for each channel
        hist_b = cv2.calcHist([channels[0]], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([channels[1]], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([channels[2]], [0], None, [256], [0, 256])
        
        mean_hist_b += hist_b.flatten()
        mean_hist_g += hist_g.flatten()
        mean_hist_r += hist_r.flatten()
        
        num_images += 1
    
    mean_hist_b /= num_images
    mean_hist_g /= num_images
    mean_hist_r /= num_images
    
    return mean_hist_b, mean_hist_g, mean_hist_r

# Plot random image with Canny's filter
def plot_canny_edges(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detect edges using Canny
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    
    # Plot the original image and edges
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edges')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()