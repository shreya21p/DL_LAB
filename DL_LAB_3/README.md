
# Lab 3 CIFAR-10 Image Classification with Convolutional Neural Networks (CNNs)

## Overview
This notebook provides a comprehensive exploration of Convolutional Neural Networks (CNNs) for image classification using the CIFAR-10 dataset. It covers fundamental CNN concepts such as convolutional layers, pooling layers, and model construction, training, and evaluation. The notebook includes tasks for understanding the impact of various hyperparameters and network architectures.

## Table of Contents
1.  [Setup and Data Loading](#setup-and-data-loading)
2.  [Data Exploration](#data-exploration)
3.  [Convolutional Layer Concepts](#convolutional-layer-concepts)
    *   [Kernel Size Comparison](#kernel-size-comparison)
    *   [Hyperparameter Study (Stride & Padding)](#hyperparameter-study-stride--padding)
    *   [Feature Map Visualization](#feature-map-visualization)
4.  [Pooling Layer Concepts](#pooling-layer-concepts)
    *   [Max Pooling vs. Average Pooling Comparison](#max-pooling-vs-average-pooling-comparison)
5.  [CNN Model Training and Evaluation](#cnn-model-training-and-evaluation)
    *   [Model Construction](#model-construction)
    *   [Training and Validation](#training-and-validation)
    *   [Evaluation Metrics and Confusion Matrix](#evaluation-metrics-and-confusion-matrix)
6.  [Additional Exercises](#additional-exercises)
    *   [Filter Comparison (16 vs. 64 Filters)](#filter-comparison-16-vs-64-filters)

## How to Use This Notebook
To run this notebook, simply execute each code cell sequentially. Ensure you have an active internet connection for downloading the CIFAR-10 dataset if it's not already cached. The notebook is designed to be self-contained.

## 1. Setup and Data Loading
This section imports necessary libraries like `numpy`, `matplotlib`, `seaborn`, `tensorflow`, and `sklearn`. It then loads the CIFAR-10 dataset, normalizes image pixel values to `[0, 1]`, and defines the class names.

## 2. Data Exploration
This section performs initial Exploratory Data Analysis (EDA) on the CIFAR-10 dataset:
*   **Sample Images**: Displays a grid of sample images from the training set with their corresponding class labels.
*   **Class Distribution**: Visualizes the distribution of classes in the training dataset using a bar plot to ensure class balance.

## 3. Convolutional Layer Concepts
This part delves into the specifics of convolutional layers.

### Kernel Size Comparison
Examines the effect of different `kernel_size` values ((3,3), (5,5), (7,7)) on the output feature map dimensions for a single convolutional layer with a sample input.

### Hyperparameter Study (Stride & Padding)
Investigates how `stride` (1, 2) and `padding` ('same', 'valid') affect the output dimensions of a convolutional layer. This helps in understanding spatial downsampling and boundary effects.

### Feature Map Visualization
Visualizes the output feature maps from a single convolutional layer applied to a sample input image. This demonstrates how filters extract different features (edges, textures, etc.) from the input.

## 4. Pooling Layer Concepts
This section compares two common types of pooling layers.

### Max Pooling vs. Average Pooling Comparison
Two simple CNN models are built, one using `MaxPooling2D` and the other using `AveragePooling2D`. Both models are trained for a few epochs, and their test accuracies are compared. The output size after each pooling operation is also printed.

## 5. CNN Model Training and Evaluation
This section builds, trains, and evaluates a complete CNN model.

### Model Construction
A sequential CNN model is defined with multiple `Conv2D` and `MaxPooling2D` layers, followed by `Flatten` and `Dense` layers for classification. The model summary is printed to show the architecture and number of parameters.

### Training and Validation
The model is compiled with the Adam optimizer and `sparse_categorical_crossentropy` loss. It is then trained on the training data for 5 epochs with a validation split of 20%. Plots show the training and validation accuracy and loss over epochs.

### Evaluation Metrics and Confusion Matrix
The trained model makes predictions on the test set. Standard classification metrics are calculated and displayed:
*   **Accuracy**
*   **Precision**
*   **Recall**
*   **F1-Score**
*   **Classification Report**: Provides per-class metrics.
*   **Confusion Matrix**: A heatmap visualization of the confusion matrix helps understand misclassifications between different classes.

## 6. Additional Exercises

### Filter Comparison (16 vs. 64 Filters)
Two simple CNN models are constructed, identical except for the number of filters (16 vs. 64) in their first convolutional layer. Both models are trained for 5 epochs, and their test accuracy and training time are compared. The analysis highlights the trade-offs between model complexity (more filters leading to more features and potentially higher accuracy) and computational cost.

## Key Findings
*   **Kernel Size**: Larger kernels capture broader features but reduce spatial dimensions more quickly.
*   **Stride & Padding**: Strides greater than 1 reduce feature map size, while 'same' padding preserves spatial dimensions.
*   **Feature Maps**: Convolutional layers learn to extract various low-level features.
*   **Pooling**: Max pooling generally outperforms average pooling for classification tasks due to its ability to retain dominant features, although the difference can be minor depending on the task and data.
*   **Model Training**: As expected, accuracy tends to increase and loss decreases over epochs.
*   **Evaluation**: The final CNN model achieves a decent accuracy on CIFAR-10, with some classes performing better than others (e.g., 'automobile', 'ship' vs. 'cat', 'dog').
*   **Filter Count**: Increasing the number of filters allows the model to learn more complex and diverse features, generally leading to higher accuracy but at the cost of increased training time and computational resources.
