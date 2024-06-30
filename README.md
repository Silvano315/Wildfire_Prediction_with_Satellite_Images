# Wildfire Detection Project

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Methods](#methods)
4. [Results](#results)
5. [References](#references)


## Introduction

Wildfires are a significant environmental hazard, causing extensive damage to ecosystems, property, and human life. The ability to predict and detect wildfires early is crucial for mitigating their impact. This project aims to harness the power of satellite imagery and deep learning techniques to develop a Convolutional Neural Network (CNN) model and Transfer Learning solution for wildfire detection and prediction.

The primary objective is to explore a satellite image dataset to build a robust Deep Learning model capable of accurately identifying wildfire occurrences. The project also considers the integration of an interpretability module with Explain like I'm 5 (ELI5), which has integrated Keras - explain predictions of image classifiers via Grad-CAM visualizations [[2](#ref2)].

This is a kind of Remote Sensing project, which conists on the acquisition of information about an object or phenomenon without making physical contact with the object. Satellite data provides valuable information for a wide range of applications including environmental monitoring, agriculture, and disaster management.

## Dataset

The dataset was taken from a Kaggle repository [[1](#ref1)]. Satellite images of areas that previously experienced wildfires in Canada.

This dataset contains satellite images (350x350px) in 2 classes :
* Wildfire : 22710 images
* No wildfire : 20140 images

The data was divided into train, test and validation with these percentages :
* Train : ~70%
* Test : ~15%
* Validation : ~15%

## Methods

### Exploratory Data Analysis (EDA)

### Deep Learning Models

### Explainability with ELI5

## Results


## References

1. <a name="ref1"></a> https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset
2. <a name="ref2"></a> https://github.com/TeamHG-Memex/eli5

