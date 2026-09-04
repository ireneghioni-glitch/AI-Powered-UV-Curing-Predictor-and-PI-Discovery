# Phase 3: The Deep Learning Core – Detailed Step‑by‑Step Guide

This guide walks you through the third phase of your MVP pipeline. By the end of this phase you will have built, trained, and evaluated a neural network regressor that takes the visual embeddings (from Phase 2) as input and predicts the %Curing Conversion (a continuous value from 0 to 100). This is where the "Deep Learning" part of your project truly comes to life.

You will learn how to:
* Understand what a neural network regressor is and how it differs from a classifier.
* Load the embeddings and prepare them for training.
* Build a Multi‑Layer Perceptron (MLP) regressor using PyTorch.
* Train the model using the training loop (forward pass, loss, backward pass, optimizer step).
* Evaluate the model on a test set.
* Save the trained model for later use in Phase 4 and Phase 5.

---

## 1. Theoretical Foundations: Neural Networks for Regression

### 1.1 What is a Neural Network?

A neural network is a computational system inspired by the structure of the human brain. It consists of interconnected layers of artificial neurons (perceptrons) that process information.

Reference: `02-what-is-a-perceptron.pdf` – Section "Perceptron"

A neuron performs a simple operation:
1. **Weighted sum**: It multiplies each input by a corresponding weight and sums them together, adding a bias term.
2. **Activation function**: The result is passed through a non‑linear function (e.g., ReLU, Sigmoid) to introduce non‑linearity.

```math
z = ∑(x_i * w_i) + b
output = activation_function(z)
```

Reference: `02-what-is-a-perceptron.pdf` – Section "Sum" and "Activation function"

### 1.2 The Multi‑Layer Perceptron (MLP)

An MLP is a feedforward neural network with one or more hidden layers between the input and output layers. The term "deep learning" refers to networks with many hidden layers.

**Why we need multiple layers**:
- A single layer (perceptron) can only learn linear relationships.
- Multiple layers with non‑linear activations allow the network to learn complex, non‑linear mappings.
- Each layer builds a more abstract representation of the input.

Reference: `03-the-deep-in-deep-learning.pdf` – Section "Forward pass"

### 1.3 Regression vs. Classification

| **Aspect** | **Classification** | **Regression** |
|------------|-------------------|----------------|
| **Output** | Discrete label (e.g., "cat" or "dog") | Continuous number (e.g., 72.3%) |
| **Output layer activation** | Softmax (probabilities) | Linear (no activation) or ReLU |
| **Loss function** | Cross‑entropy | Mean Squared Error (MSE) |
| **Evaluation metric** | Accuracy | R², RMSE, MAE |

**In your project**: You are predicting a percentage (0‑100). This is a **regression** task. The output layer has **1 neuron** with **no activation function** (linear).

Reference: `04-perceptron-with-pytorch.pdf` – Section "Applying the perceptron theory"

### 1.4 The Training Loop (How a Neural Network Learns)

The training loop consists of five essential steps, repeated for each batch of data:

1. **Forward pass**: Pass the input through the network to get a prediction.
2. **Compute loss**: Measure how far the prediction is from the true value (using a loss function like MSE).
3. **Backward pass (backpropagation)**: Compute the gradient of the loss with respect to each weight.
4. **Update weights**: Adjust the weights in the direction that reduces the loss (using an optimizer like Adam).
5. **Repeat**: Continue until the loss converges or a set number of epochs is reached.

Reference: `03-the-deep-in-deep-learning.pdf` – Section "Backward pass" and "Gradient and gradient descent"

```text
[Forward Pass] → [Compute Loss] → [Backward Pass] → [Update Weights] → [Repeat]
```

### 1.5 Why We Need a Regressor on Top of Embeddings

In Phase 2, we extracted **visual embeddings** – dense vectors (1280 numbers) that describe the structure of each molecule. However, these embeddings alone are not the final prediction.

**What we need**:
- A model that can map these embeddings to the target value (%Curing Conversion).
- A **regressor** that learns the relationship between the visual features and the conversion percentage.

**Why a separate regressor?**
- The CNN (MobileNetV2) was trained on ImageNet to recognize general visual patterns.
- The regressor is trained specifically on your data to predict the target.
- This two‑stage approach (feature extraction + regression) is more data‑efficient and less prone to overfitting than training a CNN from scratch.

Reference: `01-image-classification-theory.md` – Section "CNN Image Classification"

---

## 2. The Specification (What Phase 3 Requires)

The MVP Technical Specification (`curing-prediction-pipeline-mvp-v3-en.md`) defines Phase 3 as:

> **"The Deep Learning Core and the Training Loop"**:
> - Define a regressor architecture (Multilayer Perceptron) using PyTorch.
> - Use a training loop with forward pass, loss computation, backward pass, and optimizer step.
> - Train the model to predict the %Curing Conversion from the visual embeddings.

The specification provides a template for the regressor:

```python
class CuringPredictorNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)
```

**Key choices**:
- **Input dimension**: 1280 (the size of the embeddings from Phase 2).
- **Hidden layers**: Two hidden layers with 64 and 32 neurons.
- **Activation**: ReLU for hidden layers (adds non‑linearity).
- **Output**: 1 neuron with **no activation** (linear) for regression.
- **Loss**: Mean Squared Error (MSE).
- **Optimizer**: Adam with learning rate 0.001.

---

## 3. Step‑by‑Step Implementation

### Step 3.1: Environment Setup

Ensure you have the necessary libraries installed. In your `chemvision` environment:

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

Reference: `01-installation.pdf` – Section "Installing pytorch"

### Step 3.2: Load the Embeddings

We will load the embeddings and metadata from Phase 2.

Create a new script `phase3_train_regressor.py` inside a `phase3/` folder, with the following structure:

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).resolve().parent
DATA_P2_DIR = BASE_DIR.parent / "phase2" / "data"

INPUT_EMBED = DATA_P2_DIR / "embeddings.npy"
INPUT_META = DATA_P2_DIR / "embeddings_metadata.csv"

# ==================== LOAD DATA ====================

print("Loading embeddings...")
embeddings = np.load(INPUT_EMBED)  # shape: (208, 1280)
meta = pd.read_csv(INPUT_META)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Metadata: {len(meta)} rows")
```

**What we have**:
- `embeddings`: NumPy array of shape (208, 1280).
- `meta`: DataFrame with columns: `name`, `smiles`, `role`, `augment`.

**What we need**:
- **Input (X)**: The embeddings (1280 features per molecule).
- **Output (y)**: The target values (%Curing Conversion).

**Important**: At this point, we do not have the target values (the %Curing Conversion) because we haven't generated them yet. In a real project, these would come from experimental data. For the MVP, we will **simulate** target values for training and testing purposes.

Reference: `04-perceptron-with-pytorch.pdf` – Section "Load dataset"

### Step 3.3: Create Simulated Target Values (For MVP Demonstration)

Since we don't have experimental data, we will simulate plausible target values based on the molecule's role. This allows us to test the pipeline end‑to‑end.

**Simulation logic**:
- Type I photoinitiators: high conversion (75‑95%)
- Type II photoinitiators: medium conversion (50‑75%)
- Co‑initiators: low conversion (20‑50%)

```python
# ==================== SIMULATE TARGET VALUES ====================

print("Generating simulated target values...")
np.random.seed(42)  # For reproducibility

def simulate_conversion(role):
    if role == "PI_TypeI":
        return np.random.uniform(75, 95)
    elif role == "PI_TypeII":
        return np.random.uniform(50, 75)
    elif role == "co-initiator":
        return np.random.uniform(20, 50)
    else:
        return np.random.uniform(30, 70)  # fallback

y = np.array([simulate_conversion(role) for role in meta['role']])
print(f"Target values shape: {y.shape}")
print(f"Min: {y.min():.2f}%, Max: {y.max():.2f}%")
```

**Note**: In a real scenario, you would replace this simulation with actual experimental data.

### Step 3.4: Split Data into Train/Test Sets

We split the data into training and test sets to evaluate the model's performance on unseen data.

```python
# ==================== TRAIN/TEST SPLIT ====================

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, y, test_size=0.2, random_state=42
)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

**Why split?**
- **Train set**: Used to train the model (learn the weights).
- **Test set**: Used to evaluate the model on data it hasn't seen (measures generalization).

Reference: `04-perceptron-with-pytorch.pdf` – Section "Creating a test/train split"

### Step 3.5: Convert to PyTorch Tensors and Create DataLoaders

PyTorch uses tensors and DataLoaders for efficient batch processing.

```python
# ==================== TENSORS & DATALOADERS ====================

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)  # (N, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)    # (N, 1)

# Create datasets
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

# Create DataLoaders
BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")
```

**Why DataLoaders?**
- Efficient batching.
- Shuffling prevents the model from learning order dependencies.
- Enables parallel data loading.

Reference: `02-intro_to_computer_vision_pytorch.pdf` – Section "Batch size"

### Step 3.6: Build the Regressor Architecture

We define the neural network using PyTorch's `nn.Module`.

```python
# ==================== MODEL ARCHITECTURE ====================

class CuringPredictorNet(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, 1)  # 1 output for regression

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        return x

# Instantiate the model
input_dim = X_train.shape[1]  # 1280
model = CuringPredictorNet(input_dim)

print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
```

**Why this architecture?**
- **Input layer**: 1280 neurons (matching the embedding size).
- **Hidden layer 1**: 64 neurons (extracts high‑level features).
- **Hidden layer 2**: 32 neurons (further compresses).
- **Output layer**: 1 neuron with **linear activation** (for regression).

Reference: `curing-prediction-pipeline-mvp-v3-en.md` – Phase 3: "Defining the Regressor architecture"

### Step 3.7: Define Loss Function and Optimizer

```python
# ==================== LOSS & OPTIMIZER ====================

criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Loss: {criterion}")
print(f"Optimizer: {optimizer}")
```

**Why MSE?**
- MSE is the standard loss for regression problems.
- It penalizes large errors more heavily (squared error).
- The goal is to minimize the average squared difference between predictions and true values.

**Why Adam?**
- Adaptive learning rate.
- Combines the advantages of AdaGrad and RMSProp.
- Works well for most problems with minimal tuning.

Reference: `03-the-deep-in-deep-learning.pdf` – Section "Cost function" and "Gradient and gradient descent"

### Step 3.8: Training Loop

The training loop is the core of Phase 3. It implements the five steps described in the theory.

```python
# ==================== TRAINING LOOP ====================

def train_epoch(model, loader, criterion, optimizer):
    model.train()  # Set model to training mode
    running_loss = 0.0
    
    for batch_X, batch_y in loader:
        # 1. Zero out accumulated gradients
        optimizer.zero_grad()
        
        # 2. Forward pass
        predictions = model(batch_X)
        
        # 3. Compute loss
        loss = criterion(predictions, batch_y)
        
        # 4. Backward pass (backpropagation)
        loss.backward()
        
        # 5. Update weights
        optimizer.step()
        
        running_loss += loss.item() * batch_X.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def evaluate(model, loader, criterion):
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    
    with torch.no_grad():  # Disable gradient tracking for efficiency
        for batch_X, batch_y in loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            running_loss += loss.item() * batch_X.size(0)
    
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss
```

**Explanation of the training loop**:
1. **Zero gradients**: Clear old gradients from the previous step.
2. **Forward pass**: Pass the input through the network to get predictions.
3. **Compute loss**: Measure how wrong the predictions are.
4. **Backward pass**: Compute gradients of the loss with respect to each weight.
5. **Update weights**: Adjust weights to reduce the loss (optimizer step).

Reference: `03-the-deep-in-deep-learning.pdf` – Section "Minimizing the cost function (Details)"

### Step 3.9: Execute the Training

```python
# ==================== RUN TRAINING ====================

NUM_EPOCHS = 100
history = {"train_loss": [], "test_loss": []}

print("Starting training...")
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    test_loss = evaluate(model, test_loader, criterion)
    
    history["train_loss"].append(train_loss)
    history["test_loss"].append(test_loss)
    
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

print("Training completed!")
```

**What you should observe**:
- The training loss should decrease over time.
- The test loss should also decrease (indicating the model is generalizing).
- If the test loss starts increasing, the model is overfitting.

### Step 3.10: Evaluate the Model

After training, we evaluate the model on the test set and compute metrics.

```python
# ==================== EVALUATION ====================

def evaluate_metrics(model, loader):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in loader:
            preds = model(batch_X)
            predictions.extend(preds.numpy().flatten())
            targets.extend(batch_y.numpy().flatten())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Mean Squared Error
    mse = np.mean((predictions - targets) ** 2)
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    # R² Score
    ss_total = np.sum((targets - np.mean(targets)) ** 2)
    ss_residual = np.sum((targets - predictions) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - targets))
    
    return {
        "MSE": mse,
        "RMSE": rmse,
        "R²": r2,
        "MAE": mae,
        "predictions": predictions,
        "targets": targets
    }

# Evaluate on test set
metrics = evaluate_metrics(model, test_loader)

print("\n=== Test Set Metrics ===")
print(f"Mean Squared Error (MSE): {metrics['MSE']:.4f}")
print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
print(f"R² Score: {metrics['R²']:.4f}")
print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
```

**Interpreting the metrics**:
- **MSE**: Average of squared errors. Lower is better.
- **RMSE**: Square root of MSE. In the same units as the target.
- **R²**: Proportion of variance explained by the model. 1.0 is perfect, 0 means the model is no better than the mean.
- **MAE**: Average absolute error. Lower is better.

Reference: `02-intro_to_computer_vision_pytorch.pdf` – Section "Model Evaluation"

### Step 3.11: Visualise Results

```python
# ==================== VISUALISATION ====================

import matplotlib.pyplot as plt

# Plot loss curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["test_loss"], label="Test")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (MSE)")
axes[0].set_title("Loss Curves")
axes[0].legend()
axes[0].grid(True)

# Plot predictions vs true values
axes[1].scatter(metrics["targets"], metrics["predictions"], alpha=0.7)
axes[1].plot([0, 100], [0, 100], 'r--', label="Perfect prediction")
axes[1].set_xlabel("True Conversion (%)")
axes[1].set_ylabel("Predicted Conversion (%)")
axes[1].set_title("Predictions vs True Values")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(BASE_DIR / "training_curves.png", dpi=100)
plt.show()
print(f"Plot saved to {BASE_DIR / 'training_curves.png'}")
```

### Step 3.12: Save the Model

Save the trained model for use in Phase 4 and Phase 5.

```python
# ==================== SAVE MODEL ====================

MODEL_PATH = BASE_DIR / "curing_predictor_model.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Save model architecture metadata
import json
model_info = {
    "input_dim": input_dim,
    "hidden1": 64,
    "hidden2": 32,
    "output_dim": 1
}
with open(BASE_DIR / "model_config.json", "w") as f:
    json.dump(model_info, f, indent=2)
print("Model config saved to model_config.json")
```

---

## 4. Complete Script Code

Below is the complete script for Phase 3. Save it as `phase3/phase3_train_regressor.py`:

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).resolve().parent
DATA_P2_DIR = BASE_DIR.parent / "phase2" / "data"

INPUT_EMBED = DATA_P2_DIR / "embeddings.npy"
INPUT_META = DATA_P2_DIR / "embeddings_metadata.csv"
MODEL_PATH = BASE_DIR / "curing_predictor_model.pth"

# ==================== LOAD DATA ====================

print("Loading embeddings...")
embeddings = np.load(INPUT_EMBED)
meta = pd.read_csv(INPUT_META)

print(f"Embeddings shape: {embeddings.shape}")
print(f"Metadata: {len(meta)} rows")

# ==================== SIMULATE TARGETS ====================

print("Generating simulated target values...")
np.random.seed(42)

def simulate_conversion(role):
    if role == "PI_TypeI":
        return np.random.uniform(75, 95)
    elif role == "PI_TypeII":
        return np.random.uniform(50, 75)
    elif role == "co-initiator":
        return np.random.uniform(20, 50)
    else:
        return np.random.uniform(30, 70)

y = np.array([simulate_conversion(role) for role in meta['role']])

# ==================== TRAIN/TEST SPLIT ====================

X_train, X_test, y_train, y_test = train_test_split(
    embeddings, y, test_size=0.2, random_state=42
)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ==================== TENSORS & DATALOADERS ====================

X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==================== MODEL ====================

class CuringPredictorNet(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        return x

input_dim = X_train.shape[1]
model = CuringPredictorNet(input_dim)

# ==================== LOSS & OPTIMIZER ====================

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==================== TRAINING LOOP ====================

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_X.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            running_loss += loss.item() * batch_X.size(0)
    return running_loss / len(loader.dataset)

# ==================== RUN TRAINING ====================

NUM_EPOCHS = 100
history = {"train_loss": [], "test_loss": []}

print("Starting training...")
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    test_loss = evaluate(model, test_loader, criterion)
    history["train_loss"].append(train_loss)
    history["test_loss"].append(test_loss)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

print("Training completed!")

# ==================== EVALUATION ====================

def evaluate_metrics(model, loader):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_X, batch_y in loader:
            preds = model(batch_X)
            predictions.extend(preds.numpy().flatten())
            targets.extend(batch_y.numpy().flatten())
    predictions = np.array(predictions)
    targets = np.array(targets)
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    ss_total = np.sum((targets - np.mean(targets)) ** 2)
    ss_residual = np.sum((targets - predictions) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    mae = np.mean(np.abs(predictions - targets))
    return {"MSE": mse, "RMSE": rmse, "R²": r2, "MAE": mae, "predictions": predictions, "targets": targets}

metrics = evaluate_metrics(model, test_loader)

print("\n=== Test Set Metrics ===")
print(f"MSE: {metrics['MSE']:.4f}")
print(f"RMSE: {metrics['RMSE']:.4f}")
print(f"R²: {metrics['R²']:.4f}")
print(f"MAE: {metrics['MAE']:.4f}")

# ==================== VISUALISATION ====================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["test_loss"], label="Test")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (MSE)")
axes[0].set_title("Loss Curves")
axes[0].legend()
axes[0].grid(True)

axes[1].scatter(metrics["targets"], metrics["predictions"], alpha=0.7)
axes[1].plot([0, 100], [0, 100], 'r--', label="Perfect")
axes[1].set_xlabel("True Conversion (%)")
axes[1].set_ylabel("Predicted Conversion (%)")
axes[1].set_title("Predictions vs True")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(BASE_DIR / "training_curves.png", dpi=100)
print(f"Plot saved to {BASE_DIR / 'training_curves.png'}")

# ==================== SAVE MODEL ====================

torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

model_info = {
    "input_dim": input_dim,
    "hidden1": 64,
    "hidden2": 32,
    "output_dim": 1
}
with open(BASE_DIR / "model_config.json", "w") as f:
    json.dump(model_info, f, indent=2)
print("Model config saved to model_config.json")
```

---

## 5. Testing and Validation

### 5.1 Verify the Model Loading

After training, you can verify that the model loads correctly:

```python
# Load the saved model
model_loaded = CuringPredictorNet(input_dim=1280)
model_loaded.load_state_dict(torch.load('phase3/curing_predictor_model.pth'))
model_loaded.eval()

# Test on a single sample
sample = torch.tensor(X_test[0].reshape(1, -1), dtype=torch.float32)
prediction = model_loaded(sample)
print(f"Prediction: {prediction.item():.2f}%")
print(f"True value: {y_test[0]:.2f}%")
```

### 5.2 Interpreting the Results

| **Metric** | **Interpretation** |
|------------|-------------------|
| **Loss (MSE)** | Lower = better. Measures the average squared error. |
| **R²** | Closer to 1 = better. Indicates how much variance the model explains. |
| **RMSE** | In the same units as the target. Lower = better. |
| **MAE** | Average absolute error. Lower = better. |

**Example interpretation**:
- R² = 0.85 → The model explains 85% of the variance in the data.
- RMSE = 5.2% → The typical prediction error is about 5.2 percentage points.
- MAE = 4.1% → The average absolute error is 4.1 percentage points.

---

## 6. Troubleshooting & Tips

| **Issue** | **Solution** |
|-----------|--------------|
| **Loss not decreasing** | Try adjusting the learning rate (`lr=0.01` or `lr=0.0001`). |
| **Overfitting** | Increase the test set size, add dropout, or reduce model complexity. |
| **Underfitting** | Increase model size, train for more epochs, or adjust learning rate. |
| **Slow training** | Use a larger batch size (`BATCH_SIZE = 32` or `64`). |
| **CUDA error** | If you have a GPU, move tensors and model to `device = torch.device("cuda")`. |
| **Simulated targets unrealistic** | Replace with real experimental data when available. |

---

## 7. Summary of Phase 3 Deliverables

After completing this phase, your project folder should contain:

| **File** | **Description** |
|----------|-----------------|
| `phase3/curing_predictor_model.pth` | Trained model weights (PyTorch state dict). |
| `phase3/model_config.json` | Model architecture configuration (input_dim, layers). |
| `phase3/training_curves.png` | Visualisation of loss curves and predictions vs true values. |

You have also learned:
- What a neural network is and how it learns (forward pass, loss, backward pass, optimizer).
- How to build a regressor in PyTorch for a regression task.
- How to train and evaluate the model.
- How to interpret evaluation metrics (MSE, RMSE, R², MAE).

---

## 8. Next Steps (Preview of Phase 4)

Now that we have a trained regressor, the next phase will:

1. **Load the trained model**.
2. **Extract embeddings from Phase 2**.
3. **Create the full feature vector** by concatenating embeddings with environmental features (Is_Aqueous, LogP, PI_Concentration, UV_Dose).
4. **Train an XGBoost regressor** on the combined features.
5. **Compare the performance** of the neural network and XGBoost.

This is where the “Integration & ML” of the pipeline comes into play, as described in the specification.

Reference: `curing-prediction-pipeline-mvp-v3-en.md` – Phase 4: “Tabular Integration & Machine Learning (XGBoost)”

---

## 9. Theoretical Summary of Key Concepts

### 9.1 Forward Pass

The forward pass is the process of passing input data through the network to produce a prediction:

```text
Input → Layer 1 → Activation → Layer 2 → Activation → Output
```

### 9.2 Backward Pass (Backpropagation)

The backward pass computes the gradients of the loss with respect to each weight, propagating the error backwards through the network.

### 9.3 Loss Function (MSE)

Mean Squared Error measures the average squared difference between predictions and true values:

```math
MSE = (1/n) * Σ(prediction_i - target_i)²
```

### 9.4 Optimizer (Adam)

**Adam** (short for **Adaptive Moment Estimation**) is one of the most popular optimization algorithms used in deep learning. It was introduced by Kingma and Ba in 2014 and has become the default choice for many practitioners because it works well across a wide range of problems with minimal tuning.

#### What does an optimizer do?

In a neural network, the optimizer's job is to **update the weights** of the network in response to the gradients computed during backpropagation. The goal is to find the set of weights that minimizes the loss function.

Gradient descent tells us the direction in which to move each weight to reduce the loss. The optimizer decides **how big of a step** to take in that direction (the learning rate) and **how to adjust the step** over time.

#### Why is Adam special?

Adam combines the strengths of two other popular optimizers:

| **Optimizer** | **Strength** | **Weakness** |
|---------------|--------------|--------------|
| **SGD (Stochastic Gradient Descent)** | Simple, works well with proper tuning | Requires careful tuning of learning rate; can get stuck in plateaus |
| **AdaGrad** | Adapts learning rate per parameter; good for sparse data | Learning rate decays too aggressively over time |
| **RMSProp** | Adapts learning rate per parameter using moving averages | No momentum, can be slow in some cases |
| **Momentum** | Accelerates convergence in consistent directions | Fixed learning rate can still be problematic |

Adam combines:
1. **Momentum** (from SGD with momentum) – it keeps a moving average of past gradients to accelerate convergence in the right direction.
2. **Adaptive learning rates** (from RMSProp) – it keeps a moving average of squared gradients to scale the learning rate per parameter.

#### How does Adam work under the hood?

Adam maintains two quantities for each weight in the network:

1. **First moment (m)**: The moving average of the gradient (the mean). This acts like momentum, smoothing out oscillations and accelerating movement in consistent directions.
2. **Second moment (v)**: The moving average of the squared gradient (the uncentered variance). This is used to adapt the learning rate per parameter.

At each update step:
```math
m = β₁ * m + (1 - β₁) * g  
```
(exponential moving average of gradients)
```math
v = β₂ * v + (1 - β₂) * g²
```
(exponential moving average of squared gradients)

Where:
- `g` is the gradient of the loss with respect to the weight.
- `β₁` and `β₂` are decay rates (typical values: β₁ = 0.9, β₂ = 0.999).

Because m and v are initialized to zero, they are biased towards zero early in training. Adam applies **bias correction** to counteract this:

```math
m_hat = m / (1 - β₁ᵗ)
```
```math
v_hat = v / (1 - β₂ᵗ)
```

Finally, the weight is updated as:

```math
w = w - α * m_hat / (√v_hat + ε)
```

Where:
- `α` is the learning rate (typical: 0.001).
- `ε` is a tiny constant (e.g., 1e-8) to prevent division by zero.

#### Why Adam is a good choice for this project

| **Reason** | **Explanation** |
|------------|-----------------|
| **Works out of the box** | Requires minimal hyperparameter tuning (default values work well for most problems). |
| **Handles noisy gradients** | The momentum component smooths out fluctuations, which is useful when training on small datasets. |
| **Adaptive learning rates** | Different features (embedding dimensions) may require different learning rates; Adam automatically handles this. |
| **Fast convergence** | Adam often converges faster than plain SGD, which means fewer epochs needed. |

#### Practical considerations

- **Learning rate**: The default `lr=0.001` is a good starting point. If the loss doesn't decrease, try `0.01` or `0.0001`.
- **Weight decay**: Adam has a `weight_decay` parameter that adds L2 regularization. This can help prevent overfitting.
- **When to use other optimizers**: For very large models or when you have time to tune, SGD with momentum can sometimes achieve slightly better final performance. For most practical purposes, Adam is the safer and faster choice.

#### Reference

Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. *ICLR 2015*.

Reference: `03-the-deep-in-deep-learning.pdf` – Section "Gradient and gradient descent"

### 9.5 Evaluation Metrics

| **Metric** | **Formula** | **Goal** |
|------------|-------------|----------|
| **MSE** | (1/n) Σ(ŷ - y)² | Minimise |
| **RMSE** | √MSE | Minimise |
| **MAE** | (1/n) Σ|ŷ - y| | Minimise |
| **R²** | 1 - (SS_res / SS_tot) | Maximise (max 1) |

---

## 10. Conclusion

`phase3_train_regressor.py` successfully completes Phase 3 of the MVP pipeline, building and training a neural network regressor that maps visual embeddings to predicted %Curing Conversion. The script:

- ✅ Implements the specification exactly as described.
- ✅ Uses PyTorch for the deep learning core.
- ✅ Implements the full training loop (forward pass, loss, backward pass, optimizer step).
- ✅ Evaluates the model on a test set.
- ✅ Saves the trained model for use in Phase 4 and Phase 5.