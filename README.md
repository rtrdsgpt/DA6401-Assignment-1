# DA6401 — Assignment 1: Multi‑Layer Perceptron

#### **Aritra Dasgupta**
#### Roll No: **MA25M005**
#### M.Tech. Industrial Mathematics and Scientific Computing



## Links

| | |
|---|---|
| **GitHub Repository** | https://github.com/rtrdsgpt/DA6401-Assignment-1 |
| **W&B Report** | https://wandb.ai/ma25m005-iit-madras/DA6401%20Assignment%201/reports/DA6401-Assignment-1-Report--VmlldzoxNjEzNzMzMw?accessToken=a4wc7ev85gju0js61b84rm6tr7ywk5c9897oke7454knivoq4a7b7aer8ktdkcov |

---

Implementation of a **fully configurable Multi‑Layer Perceptron (MLP)**
from scratch using **NumPy**.
The model supports multiple optimizers, activation functions, and weight
initialization strategies and is evaluated on **MNIST and Fashion‑MNIST** datasets.

All experiments and visualizations are logged using **Weights & Biases (W&B)**.

---

## Project Structure

```text
DA6401 Assignment 1/
│
├── src/
│   ├── ann/
│   │   ├── __init__.py
│   │   ├── activations.py         # Activation functions (sigmoid, tanh, relu, softmax)
│   │   ├── neural_layer.py        # Fully connected layer implementation
│   │   ├── neural_network.py      # Main MLP model
│   │   ├── objective_functions.py # Cross‑Entropy and MSE losses
│   │   └── optimizers.py          # SGD, Momentum, NAG, RMSProp, Adam, Nadam
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_loader.py         # MNIST / Fashion‑MNIST dataset loader
│   │
│   ├── best_model.npy             # Saved best model weights
│   ├── best_config.json           # Best model configuration
│   ├── train.py                   # Model training script
│   └── inference.py               # Evaluation script
│
├── notebooks/
│   └── wandb_demo.ipynb           # W&B experiments (Q2.1 – Q2.10)
│
├── models/                        # Saved model weights
├── sweep_config.yaml              # W&B hyperparameter sweep configuration
├── requirements.txt
└── README.md
```

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Train the Model

Example training command:

```bash
python3 src/train.py \
  -d fashion_mnist \
  -e 20 \
  -b 64 \
  -l cross_entropy \
  -o adam \
  -lr 0.001 \
  -wd 0.0001 \
  -nhl 3 \
  -sz 256 \
  -a relu \
  -wi xavier \
  --no_wandb
```

---

## Run Inference

```bash
python3 src/inference.py --no_wandb
```

This evaluates the trained model on the test dataset and prints Accuracy, F1, Precision, Recall.

---

## Hyperparameter Sweep (W&B)

Run automated hyperparameter search:

```bash
wandb sweep sweep_config.yaml
wandb agent <sweep_id>
```

---

## CLI Arguments

| Flag  | Description | Default |
|-------|-------------|---------|
| `-d`  | Dataset (`mnist` or `fashion_mnist`) | `mnist` |
| `-e`  | Number of epochs | `10` |
| `-b`  | Batch size | `64` |
| `-l`  | Loss function (`cross_entropy` or `mse`) | `cross_entropy` |
| `-o`  | Optimizer (`sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam`) | `adam` |
| `-lr` | Learning rate | `0.001` |
| `-wd` | Weight decay (L2 regularization) | `0.0` |
| `-nhl`| Number of hidden layers | `3` |
| `-sz` | Hidden layer sizes | `[128]` |
| `-a`  | Activation (`sigmoid`, `tanh`, `relu`) | `relu` |
| `-wi` | Weight initialization (`random`, `xavier`) | `xavier` |

---

## Experiments (W&B)

All experiments required in Section 2 of Assignment 1 were implemented and logged using **Weights & Biases**.

Experiments performed are the following:

- Data exploration
- Hyperparameter sweeps
- Optimizer comparison
- Vanishing gradient analysis
- Dead neuron investigation
- Loss function comparison
- Global performance analysis
- Error analysis
- Weight initialization study
- Fashion‑MNIST transfer challenge

---

## Results Summary

### MNIST Best Configuration

```
Architecture : 3 Hidden Layers (128 neurons each)
Activation   : ReLU
Optimizer    : Adam
Weight Init  : Xavier
Loss         : Cross‑Entropy
Validation Accuracy ≈ 97.70%
```

### Fashion‑MNIST Best Configuration

```
Architecture : 3 Hidden Layers (256 neurons each)
Activation   : ReLU
Optimizer    : Adam
Weight Init  : Xavier
Loss         : Cross‑Entropy
Test F1      : 0.8889
Test Accuracy: 0.8894
```

### Fashion‑MNIST Transfer Experiment

| Configuration | Architecture | Optimizer | Activation | Test Accuracy |
|---------------|-------------|-----------|------------|---------------|
| Config A      | 3×256        | Adam      | ReLU       | **0.8894**    |
| Config B      | 3×128        | Adam      | ReLU       | **0.8884**    |
| Config C      | 4×128        | Nadam     | ReLU       | **0.8851**    |

---

