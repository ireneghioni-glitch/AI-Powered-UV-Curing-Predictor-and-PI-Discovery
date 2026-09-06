# Phase 3: The Deep Learning Core – Detailed Step‑by‑Step Guide
#### (Extended with Monomer Integration)

This guide walks you through the third phase of your MVP pipeline. By the end of this phase you will have built, trained, and evaluated a neural network regressor that takes **combined visual embeddings** (PI + Monomer) as input and predicts the %Curing Conversion (a continuous value from 0 to 100). This is where the "Deep Learning" part of your project truly comes to life.

You will learn how to:
* Understand what a neural network regressor is and how it differs from a classifier.
* Load **both PI and monomer embeddings** and combine them into a single dataset.
* Build a Multi‑Layer Perceptron (MLP) regressor using PyTorch.
* Train the model using the training loop (forward pass, loss, backward pass, optimizer step).
* Evaluate the model on a test set.
* Save the trained model for later use in Phase 4 and Phase 5.

---

#### 1. Theoretical Foundations: Neural Networks for Regression

##### 1.1 What is a Neural Network?
A neural network is a computational system inspired by the structure of the human brain. It consists of interconnected layers of artificial neurons (perceptrons) that process information.

*Reference: 02-what-is-a-perceptron.pdf – Section "Perceptron"*

A neuron performs a simple operation:
1. **Weighted sum**: It multiplies each input by a corresponding weight and sums them together, adding a bias term.
2. **Activation function**: The result is passed through a non‑linear function (e.g., ReLU, Sigmoid) to introduce non‑linearity.

$$z = \sum(x_i * w_i) + b$$
$$\text{output} = \text{activation\_function}(z)$$

*Reference: 02-what-is-a-perceptron.pdf – Section "Sum" and "Activation function"*

##### 1.2 The Multi‑Layer Perceptron (MLP)
An MLP is a feedforward (looking ahead to anticipate, plan for) neural network with one or more hidden layers between the input and output layers. The term "deep learning" refers to networks with many hidden layers.

Why we need multiple layers:
* A single layer (perceptron) can only learn linear relationships.
* Multiple layers with non‑linear activations allow the network to learn complex, non‑linear mappings.
* Each layer builds a more abstract representation of the input.

*Reference: 03-the-deep-in-deep-learning.pdf – Section "Forward pass"*

#### 1.3 Activation Functions: Why ReLU?

An **activation function** introduces non‑linearity into the network. Without it, multiple linear layers would be equivalent to a single linear layer – the network could not learn complex patterns like the plateau in polymerisation kinetics.

| **Function** | **Formula** | **Pros** | **Cons** |
| :--- | :--- | :--- | :--- |
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | Smooth, outputs between 0 and 1 | Vanishing gradient, slow to compute |
| **Tanh** | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | Zero‑centered, stronger gradients than Sigmoid | Still suffers from vanishing gradient |
| **ReLU** | $f(x) = \max(0, x)$ | **Fast, no vanishing gradient, induces sparsity** | Neurons can "die" (output 0 forever) |

**ReLU** stands for ***Rectified Linear Unit***.  
It is the most widely used activation function for hidden layers in modern neural networks, and it is one of the main reasons why deep learning works so well.

Here is an explanation at three levels of depth:
##### 1.3.1. The mathematical definition
ReLU is a function that **outputs 0 if the input is negative**, and the **input itself if it is positive**.

$$f(x) = max(0, x)$$

_What does this mean in practice?_  
If the neuron receives a negative value (e.g., -5) → the neuron turns off and outputs 0.  
If the neuron receives a positive value (e.g., 3) → the neuron turns on and outputs 3 (it lets the value pass through unchanged).

##### 1.3.2. Why is it called "Rectified"?
The name comes from electronics. A **rectifier** is a device that **converts alternating current** (which fluctuates up and down) **into direct current (which is only positive)**.
ReLU does exactly this with numbers: it _cuts off all negative values ​​and turns them into zero_, allowing only positive values ​​to pass through.

#### 1.3.3. Why do we use ReLU instead of Sigmoid or Tanh?
Older neural networks used **Sigmoid** or **Tanh**.  
ReLU has replaced them for three fundamental reasons that are crucial for this project:  
- It **solves the "Vanishing Gradient" problem**  
With Sigmoid: If the input is very large or very small, the curve flattens out. In those regions, the gradient (the "nudge" indicating how much to adjust the weight) is nearly zero. The network stops learning.
With ReLU: For all positive values, the gradient is always 1 (constant). The error signal travels loud and clear during backpropagation, making training much faster.  
- It is extremely **computationally efficient**  
Sigmoid requires calculating a complex exponential:  
```math
1/(1 + e^{-x})
```
ReLU requires only a simple comparison:  
```python
if x > 0: 
    return x 
else: 
    return 0
```
In the script, when processing thousands of embeddings, this speed difference is huge.
- It **introduces much-needed "non-linearity"**  
Mathematically, 
$$max(0,x)$$
max(0,x) is a non-linear function (it has a "kink" or "bend" at the origin). By placing hundreds of these "kinks" in the hidden layers, the network learns to bend the space and model any complex curve (such as the polymerization plateau).

> **The only minor drawback of ReLU** (and why it doesn't concern us):  
>ReLU has a known flaw: if a neuron consistently receives negative inputs, it always outputs 0. Its gradient becomes 0, and the neuron "dies" (it never reactivates).  
>Why doesn't this worry us? That’s because the project has two hidden layers with 128 and 64 neurons. If a neuron dies, the other 100+ neurons continue learning perfectly well. For networks of this size, ReLU is the perfect choice.

**So, why ReLU for this project?**
- **Speed**: ReLU is a simple comparison (`if x > 0`), much faster than exponentials.
- **No vanishing gradient**: For positive inputs, the gradient is always 1, so errors propagate clearly.
- **Sparsity**: Neurons that output 0 are effectively "off", making the network more efficient.

##### 1.4 Regression vs. Classification

| Aspect | Classification | Regression |
| :--- | :--- | :--- |
| **Output** | Discrete label (e.g., "cat" or "dog") | Continuous number (e.g., 72.3%) |
| **Output layer activation** | Softmax (probabilities) | Linear (no activation) or ReLU |
| **Loss function** | Cross‑entropy | Mean Squared Error (MSE) |
| **Evaluation metric** | Accuracy | $R^2$, RMSE, MAE |

In your project: You are predicting a percentage (0‑100). This is a regression task. The output layer has 1 neuron with no activation function (linear).

##### 1.4.1. Output Layer Activation
**ReLU** is applied after each hidden layer:

```python
x = self.relu(self.fc1(x))  # applies ReLU to the output of fc1
x = self.relu(self.fc2(x))  # applies ReLU to the output of fc2
return self.out(x)           # output layer is linear (no activation)
```
The output layer remains linear because you are predicting a continuous number (conversion percentage), not a probability.

##### 1.4.2. Loss Fuction
The **Mean Squared Error** measures the difference between the predicted value and the actual value.

```math
\text{MSE} = \frac{1}{n} \sum (\hat{y}_i - y_i)^2
```
If the model predicts 72.3% and the actual value is 72.3% → loss = 0.  
If the model predicts 72.3% and the actual value is 85.7% → loss = (72.3 - 85.7)² = 179.56.  

In this project: we use MSE because we are predicting a continuous value.

```python
criterion = nn.MSELoss()
```

##### 1.4.3. Evaluation Metrics
For Regression (**R²**, **RMSE**, **MAE**)
| Metric | Formula | Interpretation |
| :--- | :--- | :--- |
| **MSE** | $\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ | Mean Squared Error (penalizes large errors) |
| **RMSE** | $\sqrt{\text{MSE}}$ | Root Mean Squared Error. Average magnitude of the error or difference between predicted values and actual observed values |
| **MAE** | $\frac{1}{n} \sum_{i=1}^{n} \lvert y_i - \hat{y}_i \rvert$ | Mean Absolute. Average size of the errors between predicted values and actual values (all errors treated equally) |
| **R²** | $1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$ | Proportion of variance explained by the model |

Interpretation for this project:  
- RMSE = 5.2%: on average, your predictions are off by about 5.2 percentage points.
- MAE = 4.1%: on average, the absolute error is 4.1%.
- R² = 0.85: the model explains 85% of the data variance.

*Reference: 04-perceptron-with-pytorch.pdf – Section "Applying the perceptron theory"*

##### 1.5 The Training Loop (How a Neural Network Learns)
The training loop consists of five essential steps, repeated for each batch of data:
1. **Forward pass**: Pass the input through the network to get a prediction.
2. **Compute loss**: Measure how far the prediction is from the true value (using a loss function like MSE).
3. **Backward pass (backpropagation)**: Compute the gradient of the loss with respect to each weight.
4. **Update weights**: Adjust the weights in the direction that reduces the loss (using an optimizer like Adam).
5. **Repeat**: Continue until the loss converges or a set number of epochs is reached.

$$\text{[Forward Pass]} \rightarrow \text{[Compute Loss]} \rightarrow \text{[Backward Pass]} \rightarrow \text{[Update Weights]} \rightarrow \text{[Repeat]}$$

*Reference: 03-the-deep-in-deep-learning.pdf – Section "Backward pass" and "Gradient and gradient descent"*

##### 1.6 Why We Need a Regressor on Top of Embeddings
In Phase 2, we extracted visual embeddings – dense vectors (1280 numbers) that describe the structure of each molecule. For the extended pipeline, we have embeddings for both PIs and monomers.

What we need:
* A model that can map these embeddings (combined) to the target value (%Curing Conversion).
* A regressor that learns the relationship between the visual features of the PI, the monomer, and the conversion percentage.

Why a separate regressor?
* The CNN (MobileNetV2) was trained on ImageNet to recognize general visual patterns.
* The regressor is trained specifically on your data to predict the target.
* This two‑stage approach (feature extraction + regression) is more data‑efficient and less prone to overfitting than training a CNN from scratch.

*Reference: 01-image-classification-theory.md – Section "CNN Image Classification"*

##### 1.7 Why Combine PI and Monomer Embeddings?
The curing conversion depends on both the PI and the monomer:
* The PI generates radicals that initiate polymerisation.
* The monomer provides the double bonds that react and form the polymer network.
* The interaction between PI and monomer (e.g., compatibility, reactivity) is crucial for conversion.

By concatenating their embeddings, the regressor can learn the joint relationship between the two molecules. The final feature vector becomes:

$$\text{[1280 PI features]} + \text{[1280 Monomer features]} + \text{[Is\_Aqueous, LogP, \%PI, UV\_Dose]} = 2564\text{ features}$$

---

#### 2. The Specification (What Phase 3 Requires)
The MVP Technical Specification (curing-prediction-pipeline-mvp-v3-en.md) defines Phase 3 as:
"The Deep Learning Core and the Training Loop":
* Define a regressor architecture (Multilayer Perceptron) using PyTorch.
* Use a training loop with forward pass, loss computation, backward pass, and optimizer step.
* Train the model to predict the %Curing Conversion from the visual embeddings.

With the monomer extension, the specification is adapted as follows:
* **Input dimension**: 2560 (1280 PI + 1280 Monomer) + 4 environmental features = 2564.
* **Hidden layers**: Two hidden layers with 128 and 64 neurons (to accommodate more data).
* **Activation**: ReLU for hidden layers (adds non‑linearity).
* **Output**: 1 neuron with no activation (linear) for regression.
* **Loss**: Mean Squared Error (MSE).
* **Optimizer**: Adam with learning rate 0.001.

```python
import torch.nn as nn

class CuringPredictorNet(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)
```

---

#### 3. Step‑by‑Step Implementation

##### Step 3.1: Environment Setup
Ensure you have the necessary libraries installed. In your `chemvision` environment:
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```
*Reference: 01-installation.pdf – Section "Installing pytorch"*

##### Step 3.2: Load the Embeddings (PI + Monomer)
We will load the embeddings and metadata from Phase 2 for both PIs and monomers.
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

# PI embeddings
INPUT_PI_EMBED = DATA_P2_DIR / "embeddings.npy"
INPUT_PI_META = DATA_P2_DIR / "embeddings_metadata.csv"

# Monomer embeddings
INPUT_MONO_EMBED = DATA_P2_DIR / "embeddings_monomers.npy"
INPUT_MONO_META = DATA_P2_DIR / "embeddings_metadata_monomers.csv"

# ==================== LOAD DATA ====================
print("Loading PI embeddings...")
pi_embeds = np.load(INPUT_PI_EMBED)  # shape: (224, 1280)
pi_meta = pd.read_csv(INPUT_PI_META)

print("Loading Monomer embeddings...")
mono_embeds = np.load(INPUT_MONO_EMBED)  # shape: (40, 1280)
mono_meta = pd.read_csv(INPUT_MONO_META)

print(f"PI embeddings: {pi_embeds.shape}")
print(f"Monomer embeddings: {mono_embeds.shape}")
```

What we have:
* `pi_embeds`: NumPy array of shape (224, 1280).
* `mono_embeds`: NumPy array of shape (40, 1280).
* `pi_meta`: DataFrame with columns: name, smiles, role, augment.
* `mono_meta`: DataFrame with columns: name, smiles, role, augment.

What we need:
* A combined dataset where each row is a PI–monomer pair.
* Each pair is represented by the concatenation of the PI embedding and the monomer embedding.
* This creates a dataset of size: (224 × 40, 2560) – i.e., all possible PI–monomer combinations.

##### Step 3.3: Create Combined Dataset (PI + Monomer)
We create a dataset that pairs each PI with each monomer. This allows the model to learn the interaction between the two.

```python
# ==================== CREATE COMBINED DATASET ====================
print("Creating combined dataset...")

# Create all PI–monomer pairs
n_pis = pi_embeds.shape[0]
n_monos = mono_embeds.shape[0]

# Repeat PI embeddings for each monomer
pi_repeated = np.repeat(pi_embeds, n_monos, axis=0)  # (224*40, 1280)

# Tile monomer embeddings for each PI
mono_tiled = np.tile(mono_embeds, (n_pis, 1))        # (224*40, 1280)

# Concatenate PI + Monomer embeddings
X = np.concatenate([pi_repeated, mono_tiled], axis=1)  # (224*40, 2560)

print(f"Combined dataset shape: {X.shape}")  # (8960, 2560)
```

Why this approach?
* It creates a dataset that covers all possible combinations.
* The model will learn the interaction between each PI and each monomer.
* In a real scenario, you might have experimental data for specific combinations. Here, we simulate a full factorial design.

##### Step 3.4: Simulate Target Values (Based on PI and Monomer)
We need target values (%Curing Conversion) for each PI–monomer pair. Since we don't have experimental data, we simulate them based on chemical knowledge.

```python
# ==================== SIMULATE TARGET VALUES ====================
print("Generating simulated target values...")
np.random.seed(42)

# Function to simulate conversion based on PI and monomer
def simulate_conversion(pi_role, monomer_name, uv_dose=100):
    '''
    What does this code do?
    -----------------------
    For each PI–monomer combination, it calculates a plausible conversion value based on:
        - The type of PI (Type I → high conversion, Type II → medium, co-initiator → low).
        - The monomer factor (tri-acrylate → high, methacrylate → low).
        - The effect of the UV dose (exponential saturation).
    It returns a number between 0 and 100.
    
    IMPORTANT!
    This is really basic chemical logic for the project mvp.
    This is improvable, and so will be.'''
    # Base conversion from PI type
    if pi_role == "PI_TypeI":
        base = np.random.uniform(75, 95)
    elif pi_role == "PI_TypeII":
        base = np.random.uniform(50, 75)
    else:
        base = np.random.uniform(20, 50)
    
    # Monomer factor (functionality and type)
    monomer_factors = {
        "TMPTA": 1.0,        # tri-acrylate, high reactivity
        "DEGDA": 0.9,        # di-acrylate
        "HDDA": 0.85,
        "PEGDA": 0.8,
        "HEMA": 0.7,         # methacrylate, slower
        "MMA": 0.6,
        "Butyl acrylate": 0.75,
        "Acrylic acid": 0.7,
        "Styrene": 0.5,
        "IBOA": 0.65,
    }
    factor = monomer_factors.get(monomer_name, 0.7)
    
    # UV dose effect (saturation)
    dose_factor = 1 - np.exp(-0.01 * uv_dose)
    
    conversion = base * factor * dose_factor
    return min(max(conversion, 0), 100)

# Create target values for all combinations
# We need to know the PI role and monomer name for each row
pi_roles = pi_meta['role'].values
pi_roles_repeated = np.repeat(pi_roles, n_monos, axis=0)

monomer_names = mono_meta['name'].values
monomer_names_tiled = np.tile(monomer_names, n_pis)

y = np.array([simulate_conversion(role, name) 
              for role, name in zip(pi_roles_repeated, monomer_names_tiled)])

print(f"Target values shape: {y.shape}")
print(f"Min: {y.min():.2f}%, Max: {y.max():.2f}%")
```

Why this simulation?
* The PI type determines the radical generation efficiency.
* The monomer type determines the reactivity and maximum achievable conversion.
* The UV dose adds a saturation effect.
* This is a simplified but chemically plausible simulation for MVP demonstration.

##### Step 3.5: Split Data into Train/Test Sets
We split the combined dataset into training and test sets.

```python
# ==================== TRAIN/TEST SPLIT ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

##### Step 3.6: Convert to PyTorch Tensors and Create DataLoaders
```python
# ==================== TENSORS & DATALOADERS ====================
# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create datasets
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

# Create DataLoaders
BATCH_SIZE = 64  # Larger batch size for combined dataset
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")
```

##### Step 3.7: Build the Regressor Architecture
```python
# ==================== MODEL ARCHITECTURE ====================
class CuringPredictorNet(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64):
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

# Instantiate the model
input_dim = X_train.shape[1]  # 2560
model = CuringPredictorNet(input_dim)

print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
```

Why larger hidden layers?
* The combined dataset has more samples (8960) than the PI-only dataset (208).
* More data allows for a slightly larger model without overfitting.
* The input dimension is also larger (2560 vs 1280).

##### Step 3.8: Define Loss Function and Optimizer
```python
# ==================== LOSS & OPTIMIZER ====================
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

##### Step 3.9: Training Loop
```python
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
```

##### Step 3.10: Evaluate the Model
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
```

##### Step 3.11: Visualise Results
```python
# ==================== VISUALISATION ====================
import matplotlib.pyplot as plt

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
```

##### Step 3.12: Save the Model
```python
# ==================== SAVE MODEL ====================
import json

MODEL_PATH = BASE_DIR / "curing_predictor_model.pth"
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

model_info = {
    "input_dim": input_dim,
    "hidden1": 128,
    "hidden2": 64,
    "output_dim": 1
}
with open(BASE_DIR / "model_config.json", "w") as f:
    json.dump(model_info, f, indent=2)
print("Model config saved to model_config.json")
```

---

#### 4. Complete Script Code
Save the script below as `phase3/phase3_train_regressor.py`:

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

INPUT_PI_EMBED = DATA_P2_DIR / "embeddings.npy"
INPUT_PI_META = DATA_P2_DIR / "embeddings_metadata.csv"
INPUT_MONO_EMBED = DATA_P2_DIR / "embeddings_monomers.npy"
INPUT_MONO_META = DATA_P2_DIR / "embeddings_metadata_monomers.csv"

MODEL_PATH = BASE_DIR / "curing_predictor_model.pth"

# ==================== LOAD DATA ====================
print("Loading PI embeddings...")
pi_embeds = np.load(INPUT_PI_EMBED)
pi_meta = pd.read_csv(INPUT_PI_META)

print("Loading Monomer embeddings...")
mono_embeds = np.load(INPUT_MONO_EMBED)
mono_meta = pd.read_csv(INPUT_MONO_META)

print(f"PI embeddings: {pi_embeds.shape}")
print(f"Monomer embeddings: {mono_embeds.shape}")

# ==================== CREATE COMBINED DATASET ====================
print("Creating combined dataset...")

n_pis = pi_embeds.shape[0]
n_monos = mono_embeds.shape[0]

pi_repeated = np.repeat(pi_embeds, n_monos, axis=0)
mono_tiled = np.tile(mono_embeds, (n_pis, 1))
X = np.concatenate([pi_repeated, mono_tiled], axis=1)

print(f"Combined dataset shape: {X.shape}")

# ==================== SIMULATE TARGETS ====================
print("Generating simulated target values...")
np.random.seed(42)

def simulate_conversion(pi_role, monomer_name, uv_dose=100):
    if pi_role == "PI_TypeI":
        base = np.random.uniform(75, 95)
    elif pi_role == "PI_TypeII":
        base = np.random.uniform(50, 75)
    else:
        base = np.random.uniform(20, 50)
    
    monomer_factors = {
        "TMPTA": 1.0,
        "DEGDA": 0.9,
        "HDDA": 0.85,
        "PEGDA": 0.8,
        "HEMA": 0.7,
        "MMA": 0.6,
        "Butyl acrylate": 0.75,
        "Acrylic acid": 0.7,
        "Styrene": 0.5,
        "IBOA": 0.65,
    }
    factor = monomer_factors.get(monomer_name, 0.7)
    dose_factor = 1 - np.exp(-0.01 * uv_dose)
    conversion = base * factor * dose_factor
    return min(max(conversion, 0), 100)

pi_roles = pi_meta['role'].values
pi_roles_repeated = np.repeat(pi_roles, n_monos)
monomer_names = mono_meta['name'].values
monomer_names_tiled = np.tile(monomer_names, n_pis)

y = np.array([simulate_conversion(role, name) 
              for role, name in zip(pi_roles_repeated, monomer_names_tiled)])

print(f"Target values shape: {y.shape}")
print(f"Min: {y.min():.2f}%, Max: {y.max():.2f}%")

# ==================== TRAIN/TEST SPLIT ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
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

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ==================== MODEL ====================
class CuringPredictorNet(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64):
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
    "hidden1": 128,
    "hidden2": 64,
    "output_dim": 1
}
with open(BASE_DIR / "model_config.json", "w") as f:
    json.dump(model_info, f, indent=2)
print("Model config saved to model_config.json")
```

---

#### 5. Testing and Validation

##### 5.1 Verify the Model Loading
After training, you can verify that the model loads correctly:

```python
# Load the saved model
model_loaded = CuringPredictorNet(input_dim=2560)
model_loaded.load_state_dict(torch.load('phase3/curing_predictor_model.pth'))
model_loaded.eval()

# Test on a single sample (PI + Monomer combination)
sample = torch.tensor(X_test[0].reshape(1, -1), dtype=torch.float32)
prediction = model_loaded(sample)
print(f"Prediction: {prediction.item():.2f}%")
print(f"True value: {y_test[0]:.2f}%")
```

##### 5.2 Interpreting the Results
* `Loss (MSE)`: Lower = better. Measures the average squared error.
* `R²`: Closer to 1 = better. Indicates how much variance the model explains.
* `RMSE`: In the same units as the target. Lower = better.
* `MAE`: Average absolute error. Lower = better.

*Example interpretation:*
* $R^2 = 0.85 \rightarrow$ The model explains 85% of the variance in the data.
* $\text{RMSE} = 5.2\% \rightarrow$ The typical prediction error is about 5.2 percentage points.
* $\text{MAE} = 4.1\% \rightarrow$ The average absolute error is 4.1 percentage points.

---

#### 6. Troubleshooting & Tips

| Issue | Solution |
| :--- | :--- |
| **Loss not decreasing** | Try adjusting the learning rate ($lr=0.01$ or $lr=0.0001$). |
| **Overfitting** | Increase the test set size, add dropout, or reduce model complexity. |
| **Underfitting** | Increase model size, train for more epochs, or adjust learning rate. |
| **Slow training** | Use a larger batch size ($\text{BATCH\_SIZE} = 128$ or $256$). |
| **CUDA error** | If you have a GPU, move tensors and model to device using `device = torch.device("cuda")`. |
| **Simulated targets unrealistic** | Replace with real experimental data when available. |

---

#### 7. Summary of Phase 3 Deliverables
After completing this phase, your project folder should contain:

* `phase3/curing_predictor_model.pth`: Trained model weights (PyTorch state dict).
* `phase3/model_config.json`: Model architecture configuration (input_dim, layers).
* `phase3/training_curves.png`: Visualisation of loss curves and predictions vs true values.

You have also learned:
* What a neural network is and how it learns (forward pass, loss, backward pass, optimizer).
* How to build a regressor in PyTorch for a regression task.
* How to train and evaluate the model.
* How to interpret evaluation metrics (MSE, RMSE, $R^2$, MAE).

---

#### 8. Next Steps (Preview of Phase 4)
Now that we have a trained regressor, the next phase will:
* Load the trained model.
* Extract embeddings from Phase 2 (PI + Monomer).
* Create the full feature vector by concatenating embeddings with environmental features (Is_Aqueous, LogP, PI_Concentration, UV_Dose).
* Train an XGBoost regressor on the combined features.
* Compare the performance of the neural network and XGBoost.

This is where the "Integration & ML" of the pipeline comes into play, as described in the specification.

*Reference: curing-prediction-pipeline-mvp-v3-en.md – Phase 4: "Tabular Integration & Machine Learning (XGBoost)"*

---

#### 9. Theoretical Summary of Key Concepts

##### 9.1 Forward Pass
The forward pass is the process of passing input data through the network to produce a prediction:
$$\text{Input} \rightarrow \text{Layer 1} \rightarrow \text{Activation} \rightarrow \text{Layer 2} \rightarrow \text{Activation} \rightarrow \text{Output}$$

##### 9.2 Backward Pass (Backpropagation)
The backward pass computes the gradients of the loss with respect to each weight, propagating the error backwards through the network.

##### 9.3 Loss Function (MSE)
Mean Squared Error measures the average squared difference between predictions and true values:
$$\text{MSE} = \frac{1}{n} \sum (\text{prediction}_i - \text{target}_i)^2$$

##### 9.4 Optimizer (Adam)
Adam (short for Adaptive Moment Estimation) is one of the most popular optimization algorithms used in deep learning. It was introduced by Kingma and Ba in 2014 and has become the default choice for many practitioners because it works well across a wide range of problems with minimal tuning.

**What does an optimizer do?**
In a neural network, the optimizer's job is to update the weights of the network in response to the gradients computed during backpropagation. The goal is to find the set of weights that minimizes the loss function.

Gradient descent tells us the direction in which to move each weight to reduce the loss. The optimizer decides how big of a step to take in that direction (the learning rate) and how to adjust the step over time.

**Why is Adam special?**
Adam combines the strengths of two other popular optimizers:

| Optimizer | Strength | Weakness |
| :--- | :--- | :--- |
| **SGD (Stochastic Gradient Descent)** | Simple, works well with proper tuning | Requires careful tuning of learning rate; can get stuck in plateaus |
| **AdaGrad** | Adapts learning rate per parameter; good for sparse data | Learning rate decays too aggressively over time |
| **RMSProp** | Adapts learning rate per parameter using moving averages | No momentum, can be slow in some cases |
| **Momentum** | Accelerates convergence in consistent directions | Fixed learning rate can still be problematic |

Adam combines:
* **Momentum** (from SGD with momentum) – it keeps a moving average of past gradients to smooth out oscillations and accelerate movement in consistent directions.
* **Adaptive learning rates** (from RMSProp) – it keeps a moving average of squared gradients to scale the learning rate per parameter.

**How does Adam work under the hood?**
Adam maintains two quantities for each weight in the network:
1. **First moment (m)**: The moving average of the gradient (the mean). This acts like momentum.
2. **Second moment (v)**: The moving average of the squared gradient (the uncentered variance). This is used to adapt the learning rate per parameter.

At each update step:
$$m = \beta_1 * m + (1 - \beta_1) * g$$
$$v = \beta_2 * v + (1 - \beta_2) * g^2$$

Where:
* $g$ is the gradient of the loss with respect to the weight.
* $\beta_1$ and $\beta_2$ are decay rates (typical values: $\beta_1 = 0.9, \beta_2 = 0.999$).

Because $m$ and $v$ are initialized to zero, they are biased towards zero early in training. Adam applies bias correction to counteract this:
$$\hat{m} = \frac{m}{1 - \beta_1^t}$$
$$\hat{v} = \frac{v}{1 - \beta_2^t}$$

Finally, the weight is updated as:
$$w = w - \frac{\alpha * \hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

Where:
* $\alpha$ is the learning rate (typical: 0.001).
* $\epsilon$ is a tiny constant (e.g., $10^{-8}$) to prevent division by zero.

*Reference: Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR 2015.*

##### 9.5 Evaluation Metrics

| Metric | Goal |
| :--- | :--- |
| **MSE** | Minimise |
| **RMSE** | Minimise |
| **MAE** | Minimise |
| **R²** | Maximise (max 1) |

---

#### 10. Conclusion
`phase3_train_regressor.py` successfully completes Phase 3 of the MVP pipeline, building and training a neural network regressor that maps combined visual embeddings (PI + Monomer) to predicted %Curing Conversion.

Current status: **Ready for Phase 4 (Tabular Integration & XGBoost).**
