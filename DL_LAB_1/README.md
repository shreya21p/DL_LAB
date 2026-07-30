# Lab 1 Single Layer Perceptron

This project implements a **Single Layer Perceptron** from scratch using Python.

It covers **Banknote Authentication** first, followed by a small implementation of the Perceptron for **AND, OR and NOT gates**.

---

## Banknote Authentication

The main part of the project uses the **UCI Banknote Authentication Dataset** to classify banknotes as authentic or forged.

### Dataset

The dataset contains 4 features:

* Variance
* Skewness
* Curtosis
* Entropy

The target classes are:

* `0` – Authentic
* `1` – Forged

The dataset is downloaded directly from the UCI repository when the notebook is run.

### What is done

The notebook includes:

* Dataset exploration
* Histograms
* Boxplots
* Correlation heatmap
* Train-test split
* Feature standardization
* Custom Perceptron implementation
* Model training and prediction
* Decision boundary visualization
* Training error, weight and bias plots
* Confusion matrix
* Accuracy, Precision, Recall and F1 Score
* Learning rate comparison
* Comparison with Scikit-learn's Perceptron

---

## Logic Gates

The Perceptron is also implemented for:

* AND gate
* OR gate
* NOT gate

For these examples, the notebook shows the weight updates and the decision boundary after each update.

---

## Requirements

Python 3.x is required.

Install the required libraries using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

The notebook can be run using **Jupyter Notebook, JupyterLab or Google Colab**.

---

## How to Run

1. Open `dl_lab1_code.ipynb`.
2. Install the required libraries if they are not already installed.
3. Run the notebook **from top to bottom**.
4. The Banknote Authentication section will download the dataset automatically.
5. The results and plots will be displayed as the cells are executed.

No separate dataset download is required.

Internet connection is required for downloading the dataset.

---

## Output

The notebook generates and displays:

* Feature histograms
* Boxplots
* Correlation heatmap
* Decision boundary
* Training dynamics
* Confusion matrix
* Learning rate comparison
* AND, OR and NOT gate plots

The figures are also saved as image files, including EPS files for use in the lab report.

---

## Files

```text
dl_lab1_code.ipynb
README.md
```

The notebook contains the complete implementation and experiments.

---

## Note

The Perceptron used for the Banknote Authentication task is implemented from scratch to understand the working of the algorithm. Scikit-learn's Perceptron is used separately for comparison.
