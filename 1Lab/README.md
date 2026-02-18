# Lab 1 – Artificial Neuron (Perceptron)

## Overview

This project implements and evaluates an artificial neuron (perceptron) model for binary classification in 2D space. Two linearly separable classes of data points are generated, and weights and bias are found using random search — tested with both step and sigmoid activation functions.

---

## Project Structure

```
1Lab/
├── main.py                         # Entry point
├── generuoti_duomenis.py           # Data generation script
├── Config/
│   └── config.py                   # Path and directory configuration
├── Enums/
│   └── Activation_function.py      # Activation function enum
├── Activation_functions/
│   ├── step_activation.py          # Step activation function
│   └── sigmoid_activation.py       # Sigmoid activation function
├── Utils/
│   ├── find_weights.py             # Random search for weights and bias
│   ├── check_accuracy.py           # Accuracy checker
│   ├── perceptron.py               # Perceptron inference
│   ├── plot_results.py             # Visualization
│   └── print_verification.py       # Verification output
├── duomenys/                       # Generated CSV data
└── grafikai/                       # Generated plots
```

---

## Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Generate data

Run the data generation script first. It creates 30 two-dimensional data points split into two classes and saves them to a CSV file, and also saves a scatter plot.

```bash
python3 generuoti_duomenis.py
```

- **Class 0** – 15 points around center $(-2.0, -2.0)$, standard deviation $\sigma = 0.8$
- **Class 1** – 15 points around center $(2.0, 2.0)$, standard deviation $\sigma = 0.8$
- Random seed: `42`

### 2. Run the perceptron

```bash
python3 main.py
```

This script:
1. Loads the generated data
2. Searches for 3 valid weight sets using the **step** activation function (seed `123`)
3. Searches for 3 valid weight sets using the **sigmoid** activation function (seed `456`)
4. Saves a classification plot to `grafikai/neuronas.png`
5. Prints a verification table with accuracy for each found set

---

## Method

Weights $w_1$, $w_2$ and bias $w_0$ are found using **random search**. In each iteration, the three parameters are sampled uniformly from $[-10.0,\ 10.0]$. A candidate set $(w_1, w_2, w_0)$ is accepted if the perceptron correctly classifies all 30 data points (100% accuracy).

The search stops when 3 valid sets are found or the maximum number of iterations $N = 1{,}000{,}000$ is reached.

### Perceptron output

$$a = x_1 w_1 + x_2 w_2 + w_0$$

### Activation functions

**Step:**

$$\hat{y} = \begin{cases} 1, & a \geq 0 \\ 0, & a < 0 \end{cases}$$

**Sigmoid:**

$$\hat{y} = \text{round}\!\left(\frac{1}{1 + e^{-a}}\right)$$

---

## Results

All 6 found weight sets (3 per activation function) achieved **100% classification accuracy**.

### Step activation (seed = 123)

| Set | $w_1$ | $w_2$ | $w_0$ |
|-----|--------|--------|--------|
| 1 | 1.0263 | 4.3894 | −1.5379 |
| 2 | 9.6153 | 3.6966 | −0.3814 |
| 3 | 6.9886 | 4.4891 | 2.2205 |

### Sigmoid activation (seed = 456)

| Set | $w_1$ | $w_2$ | $w_0$ |
|-----|--------|--------|--------|
| 1 | 6.1705 | 2.5126 | 2.0823 |
| 2 | 7.7140 | 5.1823 | −6.3779 |
| 3 | 4.4668 | 3.6134 | −6.3817 |

---
