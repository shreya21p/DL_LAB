# Lab 4: Comparative Study of Deep CNN Architectures

This repository contains a Google Colab notebook (`Experiment_4_Comparative_Study_of_Deep_CNN_Architectures.ipynb`) that performs a comparative study of various Convolutional Neural Network (CNN) architectures on the CIFAR-10 dataset. The experiment explores training CNNs from scratch (LeNet-5, AlexNet) and leveraging transfer learning and fine-tuning with powerful pre-trained models (VGG16, ResNet50, MobileNetV2).



## Overview

This notebook demonstrates a comprehensive study on deep CNN architectures for image classification using the CIFAR-10 dataset. It covers:

-   **CNNs from Scratch:** Implementing and training classic architectures like LeNet-5 and AlexNet.
-   **Transfer Learning:** Utilizing pre-trained VGG16, ResNet50, and MobileNetV2 models from ImageNet as feature extractors.
-   **Fine-Tuning:** Adapting the pre-trained models by unfreezing and retraining specific layers to better suit the CIFAR-10 dataset.
-   **Hyperparameter Study:** Comparing different optimizers (Adam vs. SGD) and investigating the impact of learning rate and dense layer units.
-   **Evaluation & Comparison:** Detailed analysis of model performance using accuracy, precision, recall, F1-score, confusion matrices, and misclassified image visualizations.

## Setup and Prerequisites

To run this notebook, you will need:

-   **Google Colab Environment:** The notebook is designed to be run in Google Colab.
-   **GPU Runtime:** For faster training, it is **highly recommended** to enable a GPU runtime. In Colab, go to `Runtime > Change runtime type` and select `T4 GPU` or a similar available GPU.
-   **Libraries:** All necessary libraries (TensorFlow, Keras, NumPy, Matplotlib, Seaborn, Pandas, Scikit-learn) are imported and assumed to be available in the Colab environment.

## How to Run the Notebook

1.  **Open in Colab:** Open the `Experiment_4_Comparative_Study_of_Deep_CNN_Architectures.ipynb` file in Google Colab.
2.  **Set GPU Runtime:** Ensure your runtime is set to GPU (as described above).
3.  **Run All Cells:** Execute all cells sequentially from top to bottom. You can do this by navigating to `Runtime > Run all`.

    *   **Note:** Some training processes can take a significant amount of time, especially for deeper models or larger epoch counts. Progress will be displayed in the cell outputs.

## Experiment Sections

### 0. Setup & Imports

This section initializes the environment, imports all required Python libraries (TensorFlow, Keras, Matplotlib, etc.), sets up random seeds for reproducibility, and defines global constants such as `CLASS_NAMES`, `NUM_CLASSES`, `BATCH_SIZE`, and `IMG_SIZE`.

### Helper Utilities

Contains reusable helper functions for:
-   `default_callbacks()`: Defines `EarlyStopping` and `ReduceLROnPlateau` callbacks for model training.
-   `plot_history()`: Visualizes training history (accuracy and loss curves).
-   `evaluate_model()`: Computes and prints comprehensive evaluation metrics (accuracy, precision, recall, F1-score, classification report) and displays a confusion matrix.
-   `show_misclassified()`: Visualizes sample misclassified images from the test set.

### Task 1 — Dataset Preparation (CIFAR-10)

Loads the CIFAR-10 dataset, normalizes pixel values to `[0, 1]`, and one-hot encodes the labels. It then prints a summary of the dataset dimensions and displays 10 sample images.

### Data Pipelines

Sets up `tf.data.Dataset` pipelines for efficient data loading and preprocessing. Two pipelines are created:
-   **32×32 pipelines:** For models trained from scratch (LeNet-5, AlexNet), matching the original CIFAR-10 image size.
-   **96×96 pipelines:** For transfer learning models (VGG16, ResNet50, MobileNetV2), where images are resized to 96×96 to accommodate the pre-trained model input requirements while managing memory.

### Architecture 1 — LeNet-5 (Yann LeCun, 1998)

Implements and trains the classic LeNet-5 architecture from scratch on the 32×32 CIFAR-10 images. Includes model summary, training, history plot, evaluation, and misclassified image display.

### Architecture 2 — AlexNet (Krizhevsky et al., 2012)

Implements and trains an adapted version of AlexNet from scratch on the 32×32 CIFAR-10 images, incorporating ReLU activations, Dropout, and Batch Normalization. Includes model summary, training, history plot, evaluation, and misclassified image display.

### Task 2 & 3 — Transfer Learning with VGG16

Demonstrates transfer learning using the VGG16 model pre-trained on ImageNet. The convolutional base of VGG16 is loaded, frozen, and a new classification head (Global Average Pooling + Dense layers) is added and trained on the 96×96 CIFAR-10 images.

### Task 4 — Fine-Tuning VGG16

Extends the VGG16 transfer learning by unfreezing the last convolutional block (`block5`) of the VGG16 base and retraining the entire model with a very low learning rate. This adapts the high-level features to the CIFAR-10 dataset. A comparison plot shows the improvement from fine-tuning.

### Task 5 — Evaluate VGG16 (after fine-tuning)

Evaluates the fine-tuned VGG16 model's performance using the `evaluate_model` and `show_misclassified` helper functions.

### Additional Exercise — ResNet50 Transfer Learning & Fine-Tuning

Similar to VGG16, this section explores transfer learning and fine-tuning with ResNet50, a much deeper architecture known for its residual connections. It involves training a new head on a frozen ResNet50 base, followed by fine-tuning its last residual stage (`conv5`).

### Additional Exercise — MobileNetV2 Transfer Learning & Fine-Tuning

This section applies the same transfer learning and fine-tuning methodology to MobileNetV2, an efficient architecture designed for mobile and edge devices. It evaluates its performance characteristics, highlighting the benefits of depthwise separable convolutions.

### Additional Exercise — Adam vs SGD Optimizer Comparison

Compares the training performance (accuracy and loss curves) of two identical MobileNetV2 models (frozen base + new head) trained with different optimizers: Adam (adaptive) and SGD with Momentum (classical). This highlights the impact of optimizer choice on convergence and final accuracy.

### Additional Exercise — Learning Rate & Dense Unit Hyperparameter Study

Investigates the impact of two hyperparameters on MobileNetV2 performance (frozen base + new head):
-   **Learning Rate:** Compares `0.001` vs. `0.0001`.
-   **Dense Units:** Compares `128` vs. `256` units in the classification head's first dense layer.

### Additional Exercise — Frozen Base vs Fine-Tuned Summary

Provides a summary table and a bar chart comparing the validation accuracies achieved by each transfer learning model (VGG16, ResNet50, MobileNetV2) when using only a frozen base versus when fine-tuning specific layers. This quantifies the benefits of fine-tuning.

### Architecture Comparison Summary

This final comparison section aggregates the results of all trained models. It includes:
-   A table summarizing the historical context and key innovations of various CNN architectures.
-   A table detailing the CIFAR-10 performance (accuracy, precision, recall, F1-score, training time) of all models trained in this notebook.
-   A bar chart visualizing the test accuracies of all models.
-   A radar chart comparing accuracy, precision, recall, and F1-score across all models.

### Discussion Questions

A section with detailed answers to common discussion questions related to CNNs, transfer learning, and specific architectures (AlexNet, VGG16, Inception, ResNet, etc.).

## Key Findings

-   **Transfer Learning Dominance:** Models leveraging transfer learning (VGG16, ResNet50, MobileNetV2) consistently and substantially outperform models trained from scratch (LeNet-5, AlexNet) on CIFAR-10, demonstrating the power of pre-trained feature extractors.
-   **Benefits of Fine-Tuning:** Fine-tuning consistently provides an accuracy boost across all transfer-learned models, adapting the pre-trained features more precisely to the target dataset.
-   **Optimizer Impact:** Adam generally leads to faster convergence and often slightly higher accuracy compared to SGD with momentum for transfer learning tasks, due to its adaptive learning rates.
-   **Hyperparameter Sensitivity:** While important, the learning rate and the number of dense units in the classification head show that larger impacts typically come from the base architecture and transfer learning strategy itself.
-   **Efficiency vs. Accuracy:** MobileNetV2 achieves competitive accuracy with significantly fewer parameters and faster training times, making it suitable for resource-constrained environments, while VGG16 and ResNet50 generally achieve the highest accuracy at the cost of higher computational demands.

## Further Improvements

-   **More Extensive Hyperparameter Tuning:** Explore a wider range of learning rates, batch sizes, and dense layer configurations.
-   **Data Augmentation:** Implement more advanced data augmentation techniques to further improve model generalization, especially for models trained from scratch.
-   **Cross-Validation:** Use k-fold cross-validation for more robust performance estimates.
-   **Other Architectures:** Integrate and compare other state-of-the-art CNN architectures (e.g., EfficientNet, Vision Transformers).
-   **Quantization/Pruning:** For mobile-friendly models, explore techniques like quantization or pruning to further reduce model size and inference time.