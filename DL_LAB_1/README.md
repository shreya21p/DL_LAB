# Single Layer Perceptron for Banknote Authentication

## Overview

This project implements a **Single Layer Perceptron** from scratch using Python to classify authentic and forged banknotes from the **UCI Banknote Authentication Dataset**. The implementation follows the complete machine learning pipeline including data exploration, preprocessing, model training, evaluation, visualization, and comparison with Scikit-learn's Perceptron implementation.

The project was developed as part of a **Deep Learning Laboratory** exercise to understand the working of a perceptron without relying on machine learning libraries for the learning algorithm.

---

# Features

- Dataset loading directly from the UCI repository
- Dataset exploration and summary statistics
- Exploratory Data Analysis (EDA)
  - Histograms
  - Boxplots
  - Correlation Heatmap
- Data preprocessing
  - Train-Test Split
  - Feature Standardization
- Custom Single Layer Perceptron implementation
- Binary classification
- Decision Boundary visualization (2D)
- Training dynamics visualization
  - Training Error
  - Weight Evolution
  - Bias Evolution
- Performance Evaluation
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix
- Learning Rate Comparison
- Comparison with Scikit-learn Perceptron
- High-resolution EPS figure generation for reports/publications

---

# Dataset

## Name

**Banknote Authentication Dataset**

## Source

UCI Machine Learning Repository

Dataset URL:

https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt

---

## Description

The dataset contains statistical features extracted from images of genuine and forged banknotes.

The images were transformed using **Wavelet Transform**, and four statistical measures were extracted.

### Input Features

| Feature | Description |
|----------|-------------|
| Variance | Variance of the wavelet transformed image |
| Skewness | Asymmetry of the image distribution |
| Curtosis | Sharpness (tailedness) of the image distribution |
| Entropy | Measure of randomness or texture complexity |

### Target Variable

| Class | Meaning |
|--------|---------|
| 0 | Authentic Banknote |
| 1 | Forged Banknote |

---

## Dataset Size

- **Samples:** 1372
- **Features:** 4
- **Classes:** 2

---

# Project Workflow

```text
Load Dataset
      │
      ▼
Dataset Exploration
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Data Preprocessing
      │
      ▼
Train/Test Split
      │
      ▼
Feature Scaling
      │
      ▼
Custom Perceptron Training
      │
      ▼
Prediction
      │
      ▼
Evaluation
      │
      ▼
Visualization
      │
      ▼
Learning Rate Analysis
      │
      ▼
Comparison with Scikit-learn
```

---

# Technologies Used

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

# Requirements

Install the required Python libraries before running the project.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

# Project Structure

```text
Project
│
├── perceptron.py
├── README.md
│
├── histogram_Variance.eps
├── histogram_Skewness.eps
├── histogram_Curtosis.eps
├── histogram_Entropy.eps
│
├── feature_boxplots.eps
├── correlation_heatmap.eps
├── perceptron_decision_boundary.eps
├── training_dynamics.eps
├── confusion_matrix.eps
└── learning_rate_comparison.eps
```

---

# How to Run

## Step 1: Clone the Repository

```bash
git clone <repository-url>
```

Or download the ZIP file and extract it.

---

## Step 2: Navigate to the Project Folder

```bash
cd ProjectFolder
```

---

## Step 3: Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Step 4: Run the Program

```bash
python perceptron.py
```

or

```bash
python3 perceptron.py
```

---

# Program Execution

The program performs the following tasks in sequence.

## 1. Dataset Loading

- Downloads the Banknote Authentication Dataset directly from the UCI Machine Learning Repository.
- No manual dataset download is required.

---

## 2. Dataset Exploration

Prints:

- Dataset dimensions
- Missing values
- Statistical summary

Example:

```text
Dimensions: (1372, 5)

Missing Values

Variance    0
Skewness    0
Curtosis    0
Entropy     0
Class       0
```

---

## 3. Exploratory Data Analysis (EDA)

### Histograms

Generates one histogram for each feature:

- Variance
- Skewness
- Curtosis
- Entropy

Purpose:

- Understand feature distributions.

---

### Boxplot

Displays the spread of all numerical features.

Purpose:

- Detect outliers
- Compare feature ranges

---

### Correlation Heatmap

Displays pairwise correlations among all variables.

Purpose:

- Understand feature relationships
- Identify highly correlated variables

---

## 4. Data Preprocessing

### Feature Selection

```python
X = Features
y = Class
```

### Train-Test Split

- Training Set: **80%**
- Testing Set: **20%**
- `random_state = 42`
- Stratified sampling maintains equal class distribution.

### Feature Standardization

Uses:

```python
StandardScaler()
```

This scales all features to:

- Mean = 0
- Standard Deviation = 1

This improves convergence of the Perceptron.

---

# Custom Single Layer Perceptron

The project implements the Perceptron algorithm from scratch.

The implementation includes:

- Weight initialization
- Bias initialization
- Step activation function
- Weight update rule
- Bias update rule
- Prediction
- Training loop
- Training history storage

---

## Activation Function

```text
f(z) = 1, if z ≥ 0
f(z) = 0, otherwise
```

---

## Weight Update Rule

```text
update = learning_rate × (actual − predicted)

weights = weights + update × x

bias = bias + update
```

---

## Training History

The following values are stored after every epoch:

- Number of misclassified samples
- Weight values
- Bias value

These values are later used for visualization.

---

# Decision Boundary

A 2D decision boundary is generated using:

- Variance
- Skewness

Only two features are selected because a four-dimensional decision boundary cannot be visualized directly.

The plot includes:

- Training samples
- Testing samples
- Learned decision boundary

---

# Model Evaluation

Predictions are generated using the trained Perceptron model.

The following evaluation metrics are calculated.

## Accuracy

Overall percentage of correctly classified samples.

## Precision

Percentage of predicted positive samples that are actually positive.

## Recall

Percentage of actual positive samples correctly identified.

## F1 Score

Harmonic mean of Precision and Recall.

## Confusion Matrix

Displays:

- True Positive
- True Negative
- False Positive
- False Negative

---

# Training Dynamics

The following plots are generated.

## Training Error vs Epoch

Shows how the number of misclassified samples changes during training.

Purpose:

- Observe convergence.

---

## Weight Evolution

Displays how each weight changes across epochs.

Purpose:

- Understand parameter updates.

---

## Bias Evolution

Displays bias changes throughout training.

Purpose:

- Visualize optimization behavior.

---

# Learning Rate Comparison

The Perceptron is trained using three different learning rates.

- 0.1
- 0.01
- 0.001

The resulting training error curves are plotted together.

Purpose:

- Compare convergence speed and stability.

---

# Scikit-learn Comparison

The project trains another Perceptron using:

```python
from sklearn.linear_model import Perceptron
```

The accuracy of the Scikit-learn model is printed and compared with the custom implementation.

---

# Generated Figures

The program saves high-quality EPS figures suitable for reports and publications.

Generated files:

```text
histogram_Variance.eps
histogram_Skewness.eps
histogram_Curtosis.eps
histogram_Entropy.eps

feature_boxplots.eps

correlation_heatmap.eps

perceptron_decision_boundary.eps

training_dynamics.eps

confusion_matrix.eps

learning_rate_comparison.eps
```

Figure specifications:

- Format: EPS
- Resolution: 600 DPI
- Font: Times New Roman

---

# Expected Output

### Console Output

- Dataset summary
- Missing values
- Statistical description
- Accuracy
- Precision
- Recall
- F1 Score
- Scikit-learn Perceptron accuracy

### Graphical Output

- Histograms
- Boxplot
- Correlation Heatmap
- Decision Boundary
- Training Error vs Epoch
- Weight Evolution
- Bias Evolution
- Confusion Matrix
- Learning Rate Comparison

### Saved Files

All plots are automatically saved as high-resolution EPS images.

---

# Learning Outcomes

This project demonstrates:

- Fundamentals of the Perceptron algorithm
- Binary classification
- Data preprocessing techniques
- Feature standardization
- Exploratory Data Analysis
- Model evaluation metrics
- Decision boundary visualization
- Impact of learning rate on training
- Comparison between custom and Scikit-learn implementations
- Scientific plotting and publication-quality figure generation

---

# Notes

- Internet connectivity is required because the dataset is downloaded directly from the UCI repository.
- The custom Perceptron supports binary classification only.
- Decision boundary visualization uses only two features for easier interpretation.
- All plots are displayed during execution and also saved in EPS format.
- A fixed random seed (`random_state = 42`) ensures reproducible train-test splits and experimental results.
