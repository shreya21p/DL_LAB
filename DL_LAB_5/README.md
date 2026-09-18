# Deep Learning Lab - Experiment 5 (Lightweight Colab Version)

This notebook explores fundamental concepts in Deep Learning by training a Convolutional Neural Network (CNN) for pet breed image classification using a subset of the Oxford-IIIT Pet dataset.

## Purpose
The primary goal is to systematically study various design decisions in building and training a CNN, observing their impact on model performance (accuracy, loss, convergence).

## Experiments Covered:

1.  **Weight Initialization**: Comparing different strategies (Zero, Random Normal, Xavier/Glorot, He Normal) and their effect on training stability and convergence.
2.  **Regularization**: Investigating methods to combat overfitting (L2 Regularization, Dropout, Batch Normalization) and their influence on the gap between training and validation performance.
3.  **Batch Normalization**: Demonstrating the benefits of Batch Normalization in stabilizing training and accelerating convergence.
4.  **Optimization Algorithms**: Comparing the performance of various optimizers (SGD, Momentum, RMSProp, Adam) in terms of training speed and final accuracy.
5.  **Hyperparameter Tuning**: Analyzing the impact of learning rate, batch size, and dropout rate on model performance.
6.  **Transfer Learning**: Exploring the effectiveness of Feature Extraction versus Fine-Tuning using a pre-trained MobileNetV2 model.
7.  **5-Fold Cross-Validation**: Implementing cross-validation to obtain more robust and reliable performance estimates for different model configurations.
8.  **Final Model Evaluation**: Building and evaluating a final model on the full 37-class dataset using optimized hyperparameters, including a confusion matrix and per-class accuracy analysis.
9.  **Additional Exercises**: Exploring further configurations like Fine-Tuning with more layers and the use of Cosine Learning Rate Decay.

## Key Takeaways:

*   **He Normal initialization** is generally best for ReLU-based networks.
*   **Dropout** and **Batch Normalization** are effective regularization techniques.
*   **Adam** is often the most efficient and stable optimizer.
*   **Fine-tuning** pre-trained models with a **tiny learning rate** often yields the best results in transfer learning scenarios.
*   **Cross-validation** provides a more reliable assessment of model performance (Mean ± SD).
*   **Cosine Learning Rate Decay** can offer small, consistent improvements.

This lightweight version uses 128x128 images and 10 pet breeds for faster experimentation and reduced RAM usage, with a final evaluation on the full 37-class, 224x224 dataset.