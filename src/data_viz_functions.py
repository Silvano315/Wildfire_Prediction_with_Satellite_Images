import matplotlib.pyplot as plt
import seaborn as sns
import random
import cv2
import os
import pandas as pd


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