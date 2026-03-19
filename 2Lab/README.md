# Lab 2 – Single Sigmoid Neuron Training for Binary Classification

## Overview

This project implements a single sigmoid neuron for binary classification using the Breast Cancer Wisconsin dataset. The neuron is trained using two gradient descent methods — batch and stochastic — and experiments are conducted to evaluate the impact of learning rate, gradient descent method, and training time.

---

## Project Structure

```
2Lab/
├── main.py                         # Entry point
├── dataset/
│   ├── breast-cancer-wisconsin.data  # Original dataset
│   └── cleaned_data.csv              # Cleaned and normalized dataset
├── Utils/
│   ├── DataCleaner.py               # Data loading, cleaning, normalization
│   ├── DataSplitter.py              # Train/validation/test split
│   ├── SigmoidNeuron.py             # Sigmoid neuron with BGD and SGD
│   ├── ResultsPrinter.py            # Console output formatting
│   └── Visualizer.py                # Plot generation
└── visualizations/                   # Generated plots
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python3 main.py
```

This script:

1. Cleans the dataset (removes missing values, ID column, normalizes features)
2. Splits data into training (64%), validation (16%), and test (20%) sets
3. Trains a sigmoid neuron using batch and stochastic gradient descent
4. Evaluates both methods and prints detailed results
5. Runs learning rate experiments ($\eta = 0.01, 0.1, 0.5, 0.9$)
6. Compares training times at equal epoch counts (100, 300, 500)
7. Saves all visualizations to `visualizations/`

---

## Dataset

**Source:** [Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)

- **699** records, **9** features, **2** classes (benign = 2, malignant = 4)
- After removing **16** rows with missing values: **683** records
- Classes converted: $2 \rightarrow 0$ (benign), $4 \rightarrow 1$ (malignant)
- Features normalized to $[0, 1]$ using min-max normalization

---

## Method

### Sigmoid neuron

$$a = \sum_{k=1}^{n} w_k \cdot x_k + b$$

$$y = \sigma(a) = \frac{1}{1 + e^{-a}}$$

Classification: $\hat{y} = \text{round}(y)$, where $y \geq 0.5 \rightarrow 1$, $y < 0.5 \rightarrow 0$

### Batch gradient descent

Gradients accumulated over all samples, weights updated once per epoch:

$$w_k := w_k - \eta \cdot \frac{1}{m} \sum_{i=1}^{m} (y_i - t_i) \cdot y_i \cdot (1 - y_i) \cdot x_{ik}$$

### Stochastic gradient descent

Weights updated after each sample:

$$w_k := w_k - \eta \cdot (y_i - t_i) \cdot y_i \cdot (1 - y_i) \cdot x_{ik}$$

### Error metric

Mean squared error (MSE):

$$E(W) = \frac{1}{m} \sum_{i=1}^{m} (t_i - y_i)^2$$

---

## Results

### Default parameters ($\eta = 0.5$, 500 epochs)

| Metric              | Batch GD | Stochastic GD |
| ------------------- | -------- | ------------- |
| Training error      | 0.0354   | 0.0189        |
| Validation error    | 0.0296   | 0.0103        |
| Training accuracy   | 97.0%    | 97.9%         |
| Validation accuracy | 98.2%    | 99.1%         |
| Test accuracy       | 97.1%    | 95.6%         |
| Training time       | 3.19s    | 2.61s         |

### Learning rate comparison (validation accuracy, 500 epochs)

| $\eta$ | Batch GD | Stochastic GD |
| ------ | -------- | ------------- |
| 0.01   | 81.8%    | 99.1%         |
| 0.1    | 98.2%    | 99.1%         |
| 0.5    | 98.2%    | 99.1%         |
| 0.9    | 98.2%    | 99.1%         |

### Best model

**Stochastic GD** with $\eta = 0.5$ — highest validation accuracy (99.1%) and lowest validation error (0.0103). Test set: 131/137 correctly classified (95.6%).

---

## Key findings

- Stochastic GD converges faster and achieves lower error, but Batch GD generalizes better to test data
- Batch GD is sensitive to learning rate — fails to converge with $\eta = 0.01$ in 500 epochs
- Stochastic GD is robust to learning rate choice — achieves 99.1% across all tested values
- Training times are similar, with Stochastic GD slightly slower due to per-sample weight updates
