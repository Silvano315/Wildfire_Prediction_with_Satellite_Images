# Wildfire Detection Project

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methods](#methods)
    * [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    * [Deep Learning Models](#deep-learning-models)
    * [Explainability with ELI5](#explainability-with-eli5)
4. [Results](#results)
5. [References](#references)


## Introduction

Wildfires are a significant environmental hazard, causing extensive damage to ecosystems, property, and human life. The ability to predict and detect wildfires early is crucial for mitigating their impact. This project aims to harness the power of satellite imagery and deep learning techniques to develop a Convolutional Neural Network (CNN) model and Transfer Learning solution for wildfire detection and prediction.

The primary objective is to explore a satellite image dataset to build a robust Deep Learning model capable of accurately identifying wildfire occurrences. The project also considers the integration of an interpretability module with Explain like I'm 5 (ELI5), which has integrated Keras - explain predictions of image classifiers via Grad-CAM visualizations [[1](#ref1)].

This is a kind of Remote Sensing project, which conists on the acquisition of information about an object or phenomenon without making physical contact with the object. Satellite data provides valuable information for a wide range of applications including environmental monitoring, agriculture, and disaster management.

## Dataset

The dataset was taken from a Kaggle repository [[2](#ref2)]. Satellite images of areas that previously experienced wildfires in Canada.

This dataset contains satellite images (350x350px) in 2 classes :
* Wildfire : 22710 images
* No wildfire : 20140 images

The data was divided into train, test and validation with these percentages :
* Train : ~70%
* Test : ~15%
* Validation : ~15%

## Methods

### Exploratory Data Analysis (EDA)

In the exploratory data analysis (EDA) phase of this project, I began by verifying the dataset split percentages and distribution to ensure everything was consistent with the details provided by the Kaggle dataset owner. Our checks confirmed that the dataset splits were accurate and properly distributed.

Next, I checked for corrupted files and inconsistent image dimensions. This step was crucial to maintain data integrity and quality, and I identified and removed two corrupted images from the dataset.

Given that the image filenames were composed of longitude and latitude coordinates, I leveraged this information for a comprehensive geographical analysis. I extracted these coordinates and visualized them on a map, which I saved as an HTML file using the **Folium** package. You can view this interactive map [here](Saved_Images/earth_heatmap.html) or you can see what should appear [here](Saved_Images/html_map.png). Additionally, I used the **Geopandas** package to create a heatmap, illustrating the geographical distribution of the images on an Earth base map. This heatmap can be viewed [here](Saved_Images/earth_heatmap.html).

To assess the quality of the images, I conducted an in-depth analysis focusing on various aspects. For brightness and contrast, I used boxplots to analyze the data for each class and split, providing insights into the overall image quality and consistency across different classes. I measured edge density using Canny's filter, chosen for its effectiveness in detecting a wide range of edges in images. For sharpness, I employed the Laplacian operator, which is ideal due to its sensitivity to regions of rapid intensity change.

Furthermore, I explored data enhancement techniques, including sharpening and noise reduction, to improve image quality. I also performed data augmentation with various transformations, such as rotation, width and height shifts, zoom, and flips, to increase the diversity of the training dataset. To ensure these augmentations retained key features while introducing necessary variability, I visualized the augmented images. An example of an augmented image can be seen [here](Saved_Images/Augmented_Images.png).

### Deep Learning Models

In the deep learning models phase of this project, I applied data augmentation techniques to enhance the training set. I utilized various augmentation strategies, including rotation, width and height shifts, shear transformations, zoom, and horizontal flips, ensuring that the augmented data closely resembled real-world variations while maintaining the essential features of the original images.

For model building, I explored two distinct approaches. The first approach involved constructing a **Convolutional Neural Network (CNN)** from scratch. This model was designed with multiple convolutional and pooling layers, followed by dense layers with dropout and L1 penalty for regularization.

In addition to the CNN model, I implemented a transfer learning approach using the **InceptionV3 model** [[3](#ref3)], pre-trained on the ImageNet dataset. This method leveraged the powerful feature extraction capabilities of InceptionV3, enabling the model to learn from the pre-existing knowledge while adapting to the specific task of wildfire detection. I added custom layers on top of the InceptionV3 base to fine-tune the model for this binary classification task.

Training the models involved using a combination of early stopping and model checkpointing on validation accuracy to prevent overfitting and ensure that the best-performing model was saved. 


### Explainability with ELI5

## Results


## References

1. <a name="ref1"></a> https://github.com/TeamHG-Memex/eli5 
2. <a name="ref2"></a> https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset
3. <a name="ref3"></a> https://arxiv.org/abs/1512.00567
