# CS3807 Deep Learning Laboratory - Experiment 2

**Implementation of a Multi-Layer Perceptron (MLP) for Multi-Class Image Classification & XOR Perceptron Analysis**

This repository contains the source code, generated plots, and LaTeX report for Experiment 2 of the CS3807 Deep Learning Laboratory course. 

## Table of Contents
* **Project Overview**
* **Repository Structure**
* **Dependencies**
* **Usage & Execution**
* **Results Summary**

---

## Project Overview

This project is divided into two primary tasks:

1. **Fashion-MNIST MLP Classifier:** Building, training, and evaluating a Multi-Layer Perceptron (MLP) to classify 10 distinct categories of clothing using the Fashion-MNIST dataset. It includes automated hyperparameter optimization using `RandomizedSearchCV` via the `SciKeras` wrapper to find the optimal model configuration.
2. **XOR Gate Perceptron Analysis:** A from-scratch implementation of the Perceptron Learning Algorithm attempting to solve the XOR logic gate. This includes a mathematical and visual analysis (K4 level) of why a single-layer perceptron fails to converge on linearly inseparable data, demonstrating the "wiper effect" oscillation.

---

## Repository Structure

* `fashion_mnist_mlp.py`: Main Python script for data preprocessing, baseline MLP construction, model evaluation, and hyperparameter tuning.
* `xor_perceptron.py`: Python script demonstrating the perceptron learning algorithm failing to converge on the XOR gate.
* `Experiment_2-v2.tex`: The complete LaTeX source code for the lab report.
* `plots/`: Directory containing all generated visualizations (e.g., `pixel.png`, `cm.png`, `xor.png`).
* `README.md`: Project documentation.

---

## Dependencies

Ensure you have Python 3.8+ installed. The following libraries are required to run the code:

* `tensorflow`
* `scikit-learn` (Version < 1.6.0 recommended for SciKeras compatibility)
* `scikeras`
* `numpy`
* `matplotlib`
* `seaborn`

You can install the dependencies using pip:

```bash
pip install tensorflow "scikit-learn<1.6.0" scikeras numpy matplotlib seaborn
