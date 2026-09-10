# Outputs from `train_regressor.py`
## 1st version output
```python
'''
Future improving:
-----------------
Partition code in modouls as best practice
(as described in optimal_code_partitioning.md)
'''

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

from model import CuringPredictorNet


# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).resolve().parent
# data dir path in this sub-folder
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
# visuals dir path in this sub-folder
VISUALS = BASE_DIR / "visuals"
VISUALS.mkdir(parents=True, exist_ok=True)
# saved model path in this sub-folder
MODEL_PATH = BASE_DIR / "curing_predictor_model.pth"

# folder in which embeddings and metadata files we need to access to 
# are located (in phase2/)
DATA_P2_DIR = BASE_DIR.parent / "phase2" / "data"

# PIs
INPUT_PI_EMBED = DATA_P2_DIR / "embeddings_PIs.npy"
INPUT_PI_META = DATA_P2_DIR / "embeddings_metadata_PIs.csv"

# monomers
INPUT_MONO_EMBED = DATA_P2_DIR / "embeddings_monomers.npy"
INPUT_MONO_META = DATA_P2_DIR / "embeddings_metadata_monomers.csv"


# ==================== LOAD DATA ====================

print("Loading of PIs embeddings...")
pi_embeds = np.load(INPUT_PI_EMBED)             # NumPy array of shape (224, 1280)
pi_meta = pd.read_csv(INPUT_PI_META)            # DataFrame with columns: name, smiles, role, augment

print("Loading of Monomers embeddings...")
mono_embeds = np.load(INPUT_MONO_EMBED)         # NumPy array of shape (40, 1280)
mono_meta = pd.read_csv(INPUT_MONO_META)        # DataFrame with columns: name, smiles, role, augment

print("PIs and Monomers embeddings loaded successfully.")
print(f"    PIs embeddings: {pi_embeds.shape}")
print(f"    Monomers embeddings: {mono_embeds.shape}")


# ==================== COMBINE DATASETS ====================
'''
The two datasets must be merged into a single one:
    - each row is about a different PI-monomer pair (all possible combinations);
    - each pair represented by concatenation of PI embedding and monomer embedding.

This approach:
    - creates a dataset that covers all possible combinations.
    - makes the model learn the interaction between each PI and 
      each monomer.
In a real scenario, you're likely to have experimental data for specific combinations.
Here, we simulate a full factorial design.
'''
print("Creating the combined dataset...")

# 1. create all possible PI-monomer pairs
n_pis = pi_embeds.shape[0]      # n_pis = 224 (`reps`)
n_monos = mono_embeds.shape[0]  # n_monos = 40 (`repeats`)
# 224 × 40 = 8.960 possible combinations

'''
In the resulting combined dataset, we want each row to represent 
a (PI, Monomer) pair. 
To do this, we need to create two matrices with the same shape (8960, 1280) 
but different ordering:

    1. `pi_repeated` – for the PIs
        You want the first PI to appear for all 40 monomers, 
        then the second PI for all 40 monomers, and so on.
        How do we get it? Using `np.repeat(a, repeats, axis=0)`:
            repeats every single line of 'a' for 'repeats' times, one after another.
    2. `mono_tiled` – for the monomers
        You want the complete sequence of monomers (1, 2, 3, ..., 40) 
        to be repeated for each PI.
        How do we get it? Using `np.tile(a, (reps, 1))`:
            Repeats the entire block `a` for `reps` times along axis 0.'''

# 2. Repeat PI embeddings for each monomer
pi_repeated = np.repeat(pi_embeds, n_monos, axis=0)     # (224*40, 1280)

# 3. Tiling monomer embeddings for each PI
mono_tiled = np.tile(mono_embeds, (n_pis, 1))           # (224*40, 1280)

# 4. concatenate PI + monomer embeddings
X = np.concatenate([pi_repeated, mono_tiled], axis=1)   # (224*40, 2560)

print(f"Combined dataset shape: {X.shape}")             # (8960, 2560)


# ==================== SIMULATE TARGET VALUES ====================
'''
We need target values (%Curing Conversion) for each PI–monomer pair. 
Since we don't have experimental data, we simulate them based on chemical knowledge.
'''
print("Generating simulated target values...")
np.random.seed(42)

def monomer_factor():
    '''
    Monomer reactivity factors (relative scale from 0.0 to 1.0)
    Based on known principles of radical polymerisation kinetics:
      - Functionality: tri-acrylates (3 double bonds) > di-acrylates > mono-acrylates > methacrylates
      - Steric effects: bulky groups (IBOA) or methyl groups (methacrylates) reduce reactivity
      - Aromatic systems (Styrene) exhibit slower kinetics
    These factors are chemically plausible estimates for MVP demonstration.
    TMPTA (tri-acrylate) is set as reference (1.0), Styrene as lowest (0.5).
    In production, these would be replaced by experimental data or QSAR-derived values.
    '''
    monomer_factors_dict = {
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
    return monomer_factors_dict

'''
IMPORTANT!

Future Improvement: QSAR-based Reactivity Prediction
----------------------------------------------------
The current monomer factors are static estimates based on chemical intuition.
This approach will be replaced with a Quantitative Structure-Activity Relationship (QSAR) model.

Planned implementation:
  1. Compute molecular descriptors (e.g., logP, polar surface area, HOMO/LUMO energies)
     using RDKit's descriptor calculation capabilities.
  2. Use these descriptors to predict reactivity factors via a trained regression model.
  3. This will allow dynamic reactivity estimation for any monomer, including new candidates,
     without manual factor assignment.

RDKit descriptors to explore:
  - Number of rotatable bonds (steric flexibility)
  - Polar surface area (polarity effects)
  - Molecular weight (diffusion effects)
  - HOMO/LUMO energies (radical stability)
  - Double bond count (functionality)
  - Electronegativity descriptors (electronic effects)

This will transform the current rule-based simulation into a data-driven,
generalizable reactivity prediction system.'''

# Function for simulating conversion based on PI and mnonomer
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
    This is a simplified but chemically plausible simulation for MVP demonstration.
    This can absolutely be improved and optimized, and so it will be.
    '''
    # Base conversion from PI type
    if pi_role == "PI_TypeI":
        base = np.random.uniform(75,95)
    elif pi_role == "PI_TypeII":
        base = np.random.uniform(50, 75)
    else:
        base = np.random.uniform(20, 50)

    # Monomer factor (functionality and type)
    '''
    The factors represent a scale of relative monomer reactivity, 
    based on known chemical principles of radical polymerization.
    '''
    monomer_factors = monomer_factor()
    factor = monomer_factors.get(monomer_name, 0.7)
    '''
    0.7 is the default (fallback) value assigned to the reactivity 
    factor if the monomer name is not found in the `monomer_factors` 
    dictionary. 
    It is a "safe" and "average" value for unknown or as-yet-undefined 
    monomers.
    
    IMPORTANT!
    Will be removed once molecular descriptors (RDKit) are introduced: 
    reactivity factors will then be dynamically calculated from molecular 
    structure (QSAR approach), making this manual dictionary and its 
    fallback obsolete. 
    The function will also be revised to accept SMILES instead of names.
    '''

    # UV dose effect
    dose_factor = 1- np.exp(-0.01 * uv_dose)
    conversion = base * factor * dose_factor
    return min(max(conversion, 0), 100)

# Create target values for all combinations
# PI role and monomer name for each row are mandatory infos

# PI
pi_roles = pi_meta['role'].values
pi_roles_repeated = np.repeat(pi_roles, n_monos, axis=0)

# monomer
monomer_names = mono_meta['name'].values
monomer_names_tiled = np.tile(monomer_names, n_pis)

y = np.array([simulate_conversion(role, name)
              for role, name in zip(pi_roles_repeated, monomer_names_tiled)])

print(f"Target values shape: {y.shape}")
print(f"Min: {y.min():.2f}%, Max: {y.max():.2f}%")


# ==================== TRAIN/TEST SPLIT ====================
'''
We split the combined dataset into training and test sets.

Current split: Train (80%) + Test (20%) — simplified for MVP.
In future, a Validation set will be added for early stopping and hyperparameter tuning
(e.g., Train 70%, Validation 15%, Test 15%) to follow standard ML best practices.
'''
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset successfully splitted for training and testing.")
print(f"    Train set: {X_train.shape[0]} samples")
print(f"    Test set: {X_test.shape[0]} samples")


# ==================== TENSORS & DATALOADERS ====================
'''
NumPy data is "raw" and not optimized for deep learning.
We convert them into PyTorch data structures designed 
to be efficient during training.

PyTorch trains models using mini-batches (groups of samples) 
for two reasons:
    - Efficiency: calculating the gradient on an entire batch 
      (e.g., 64 samples) is faster than doing so one sample at 
      a time (GPUs are optimized for vector operations).
    - Stability: the gradient calculated on a batch is less 
      "noisy," and convergence is more stable.
This code block prepares the data for batch processing.

Tensors can be moved to GPUs, support autograd (automatic 
gradient calculation), and are optimized for mathematical operations.
'''
# convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
# Specify that the values ​​are 32-bit floating-point numbers 
# (the standard for deep learning).
'''
Reshape the tensor. 
-1 means "automatically infer the dimension," and 1 means 
"one column." Thus, an array with shape (N,) (a vector) 
becomes (N, 1) (a matrix with one column). Why? PyTorch expects the 
model output to have the shape (batch_size, output_dim). 
If output_dim = 1, it must be an (N, 1) tensor, not (N,).'''
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# create datasets
'''
It wraps the (X, y) pairs into a single object. 
When the dataset is iterated, it returns an (input, target) tuple. 
This makes the code cleaner and more manageable.'''
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

# create DataLoaders
'''
Each batch will contain 64 samples. 
The training dataset (7,168 samples) will be 
divided into ceil(7,168/64) = 112 batches.'''
BATCH_SIZE = 64
'''
64 is a common compromise between speed and stability. 
It is large enough to leverage GPUs (for efficient 
vector calculations) and reduce gradient noise, yet 
small enough to fit in memory and provide frequent updates. 
It is an empirical value that works well for medium-sized 
datasets (~9,000 samples).'''
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
'''
Important for training: shuffle the samples at each epoch. 
This prevents the model from learning the order of the data 
(e.g., if the data is sorted by PI type, the model might learn 
to "predict" based on the order rather than the features).'''
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
'''
For the test, the order does not matter because there is no 
training, and maintaining the order aids reproducibility.'''
# Shuffling: Solves model bias, memorization, and poor convergence 
# by optimizing model accuracy and loss trajectory.

print(f"Train batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")


# ==================== MODEL ARCHITECTURE ====================
# model initiation
# shape[1] is the number of features per sample 
# → 2560 (1280 PI + 1280 Monomer).
input_dim = X_train.shape[1]    # 2560
model = CuringPredictorNet(input_dim)
print(model)
'''
Calculate the total number of trainable 
parameters (weights + biases) of the model. 
p.numel() returns the number of elements 
in each parameter tensor.'''
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# ==================== LOSS & OPTIMIZER ====================
criterion = nn.MSELoss() # Mean Squared Error Regression
optimizer = optim.Adam(model.parameters(), lr=0.01)
'''
model.parameters() - It tells the optimizer which weights to update.
lr=0.01 - Learning rate. It controls the size of the steps Adam takes 
          toward the minimum of the loss function.
'''

# ==================== TRAINING LOOP ====================
'''
The training loop executes one training epoch:
"Take all the samples from the dataset (divided into 
batches), pass them through the model, calculate the 
error, and adjust the weights to reduce the error."
'''
# An epoch is a complete pass of ALL training data through 
# the model, from the first sample to the last.
def train_epoch(model, loader, criterion, optimizer):
    # put the model into training mode
    model.train()
    # Initialize a variable to accumulate the total 
    # loss across all batches. 
    # At the end, we will divide it by the number of 
    # samples to obtain the average loss for the epoch.
    running_loss = 0.0
    # loader is the created DataLoader
    for batch_X, batch_y in loader:
        # Resets the accumulated gradients in the model to zero.
        # In PyTorch, gradients are accumulated (summed) with each 
        # call to `loss.backward()`. 
        # If we did not zero them out, the gradients from the previous 
        # batch would be added to those of the current batch, leading 
        # to incorrect results.
        optimizer.zero_grad()
        # Pass the input batch through the neural network (forward pass). 
        # The forward() method you defined in the CuringPredictorNet 
        # class is executed automatically.
        predictions = model(batch_X)
        # Calculate the loss (error) by comparing the predictions 
        # with the actual values ​​(batch_y), using the loss function 
        # previously defined (nn.MSELoss()).
        loss = criterion(predictions, batch_y)
        # Backpropagation. It calculates the loss gradients with respect 
        # to all model weights (and stores them in the parameters, alongside 
        # the weights themselves).
        # Each parameter (weight and bias) of fc1, fc2, and out has a .grad 
        # attribute that now holds the calculated gradient.
        loss.backward()
        # Update the model weights using the gradients calculated in 
        # `loss.backward()`. The optimizer (Adam) determines how much to adjust 
        # each weight (learning rate, momentum, etc.).
        # Result: The weights are modified, and the model "learns" from this batch.
        optimizer.step()
        # By multiplying the average batch loss by the number of samples, 
        # we get the total loss for this batch.
        running_loss += loss.item() * batch_X.size(0)
    # Divide the total loss by the total number of samples in the dataset 
    # (len(loader.dataset)).
    # Result: The average loss for the entire epoch.
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in loader:
            predictions = model(batch_X)
            loss= criterion(predictions, batch_y)
            running_loss += loss.item() * batch_X.size(0)
        return running_loss / len(loader.dataset)\


# ==================== RUN TRAINING ====================
'''
NUM_EPOCHS is set to 100 as a fixed empirical value for MVP simplicity.
With synthetic data, convergence typically occurs well before 100 epochs,
making this a safe and practical choice for demonstration purposes.

In future production versions, this will be replaced by Early Stopping
with a validation set to automatically determine the optimal number
of epochs and prevent overfitting.
'''
NUM_EPOCHS = 100
history = {
    "train_loss": [], 
    "test_loss": []
}

print("Strating training...")
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    test_loss = evaluate(model, test_loader, criterion)
    history["train_loss"].append(train_loss)
    history["test_loss"].append(test_loss)
    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

print("Training complete.")


# ==================== MODEL EVALUATION ====================
'''
This function:
    1. Passes all test set samples through the model 
       (without updating the weights).
    2. Collects all predictions and actual values ​​into two lists.
    3. Calculates 4 metrics to evaluate model performance:
        - MSE (Mean Squared Error)
        - RMSE (Root Mean Squared Error)
        - R² (Coefficient of determination)
        - MAE (Mean Absolute Error)
    4. Returns everything in a dictionary.
'''
def evaluate_metrics(model, loader):
    # put model in evaluation mode
    model.eval()
    predictions = []
    targets = []
    # disable gradient calculation 
    # not needed, and saves memory/improves speed
    with torch.no_grad():
        for batch_X, batch_y in loader:
            # Forward pass on batch_X
            preds = model(batch_X)
            # convert the PyTorch tensor into a NumPy array (numpy()), 
            # transform the tensor from (batch_size, 1) to a 
            # 1D array of shape (batch_size,) (flatten()) and add the 
            # elements to the `predictions` list.
            predictions.extend(preds.numpy().flatten())
            # same on real values from test batch
            targets.extend(batch_y.numpy().flatten())
    # Convert lists into NumPy arrays to perform efficient vector calculations
    predictions = np.array(predictions)
    targets = np.array(targets)
    # mean of the squared errors
    mse = np.mean((predictions - targets) ** 2)
    # square root of MSE
    rmse = np.sqrt(mse)
    # Total Sum of Squares -  measures the total variance of the data
    # the extent to which actual values ​​are dispersed around their mean
    ss_total = np.sum((targets - np.mean(targets)) ** 2)
    # Sum of Squared Residuals - measures the model's error
    # the extent to which predictions deviate from the actual values
    ss_residual = np.sum((targets - predictions) ** 2)
    # proportion of variance explained by the model
    # 1 = perfect, 0 = the model explains nothing
    r2 = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    # mean absolute error
    mae = np.mean(np.abs(predictions - targets))
    return {
        "MSE": mse, 
        "RMSE": rmse, 
        "R²": r2, 
        "MAE": mae, 
        "predictions": predictions, 
        "targets": targets
    }

metrics = evaluate_metrics(model, test_loader)

print("\n=== Test Set Metrics ===")
print(f"MSE: {metrics['MSE']:.4f}")
print(f"RMSE: {metrics['RMSE']:.4f}")
print(f"R²: {metrics['R²']:.4f}")
print(f"MAE: {metrics['MAE']:.4f}")


# ==================== VISUALISATION ====================
'''
This function creates a two-panel plot that allows for a visual assessment 
of the model's performance:
    - Left panel: the progression of the loss (error) during training and 
      on the test set, epoch by epoch.
    - Right panel: a direct comparison between the actual values ​​and those 
      predicted by the model.
It helps determine whether the model is learning effectively, whether overfitting 
is occurring and how accurate the predictions actually are.
'''
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# 1 row, 2 columns

# plot the loss on the training/test set for each epoch
axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["test_loss"], label="Test")
# assign labels to the axes
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (MSE)")
# set other plot features
axes[0].set_title("Loss Curves")
axes[0].legend()
axes[0].grid(True)

''''
Create a scatter plot

Each point represents a sample from the test set:
    - X: the actual value (target) → metrics["targets"];
    - Y: the value predicted by the model → metrics["predictions"].
'''
axes[1].scatter(metrics["targets"], metrics["predictions"], alpha=0.7)
# draw a dashed red diagonal line from (0,0) to (100,100)
axes[1].plot([0, 100], [0, 100], 'r--', label="Perfect")
axes[1].set_xlabel("True Conversion (%)")
axes[1].set_ylabel("Predicted Conversion (%)")
axes[1].set_title("Predictions vs True")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(BASE_DIR / VISUALS / "training_curves.png")
print(f"Plot saved to {BASE_DIR / VISUALS / 'training_curves.png'}")


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


'''
Since this script is designed to be executed directly, 
the `if __name__ == "__main__"` block is not mandatory.

If in the future you want to import functions from this 
script into another file (e.g. for inference in Phase 4), 
it would be best to protect the main code with the 
`if __name__ == "__main__"` block.
'''
```

Gave this output:

```text
Loading of PIs embeddings...
Loading of Monomers embeddings...
PIs and Monomers embeddings loaded successfully.
    PIs embeddings: (208, 1280)
    Monomers embeddings: (40, 1280)
Creating the combined dataset...
Combined dataset shape: (8320, 2560)
Generating simulated target values...
Target values shape: (8320,)
Min: 6.33%, Max: 59.88%
Dataset successfully splitted for training and testing.
    Train set: 6656 samples
    Test set: 1664 samples
Train batches: 104
Test batches: 26
CuringPredictorNet(
  (fc1): Linear(in_features=2560, out_features=128, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (out): Linear(in_features=64, out_features=1, bias=True)
)
Total parameters: 336129
Strating training...
Epoch   1/100 | Train Loss: 103.0694 | Test Loss: 36.7467
Epoch  10/100 | Train Loss: 17.1306 | Test Loss: 15.3095
Epoch  20/100 | Train Loss: 16.0473 | Test Loss: 15.4846
Epoch  30/100 | Train Loss: 13.7837 | Test Loss: 14.2882
Epoch  40/100 | Train Loss: 12.7625 | Test Loss: 15.1638
Epoch  50/100 | Train Loss: 11.8702 | Test Loss: 16.9798
Epoch  60/100 | Train Loss: 11.1255 | Test Loss: 15.6981
Epoch  70/100 | Train Loss: 10.6101 | Test Loss: 16.0240
Epoch  80/100 | Train Loss: 9.4718 | Test Loss: 16.7457
Epoch  90/100 | Train Loss: 9.1493 | Test Loss: 17.9678
Epoch 100/100 | Train Loss: 9.0372 | Test Loss: 21.7829
Training complete.

=== Test Set Metrics ===
MSE: 21.7829
RMSE: 4.6672
R²: 0.8368
MAE: 3.7423
Plot saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\visuals\training_curves.png
Model saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\curing_predictor_model.pth
Model config saved to model_config.json
```
The The model is **overfitting** (memorizing the training data instead of generalizing).  
The loss on the test set started to rise again after epoch 30, while the training loss continued to decrease. This is a classic example of overfitting: the model learned the training data perfectly but lost the ability to generalize.

<div style="border-left: 4px solid #8250df; background-color: rgba(130, 80, 223, 0.1); padding: 10px 15px; margin: 10px 0;">
    <strong style="color: #8250df;">NEXT FIX:</strong><br> 
    <b>Set <code>NUM_EPOCHS = 30</code> in the RUN TRAINING section</b> and run it again. There should be an immediate improvement.
</div>

To do this properly (without looking at the test set during training), we should implement a validation set and use it to decide when to stop. However, **for the MVP, we will simply reduce the number of epochs to 30 for now**.

## 2nd version output
```text
Loading of PIs embeddings...
Loading of Monomers embeddings...
PIs and Monomers embeddings loaded successfully.
    PIs embeddings: (208, 1280)
    Monomers embeddings: (40, 1280)
Creating the combined dataset...
Combined dataset shape: (8320, 2560)
Generating simulated target values...
Target values shape: (8320,)
Min: 6.33%, Max: 59.88%
Dataset successfully splitted for training and testing.
    Train set: 6656 samples
    Test set: 1664 samples
Train batches: 104
Test batches: 26
CuringPredictorNet(
  (fc1): Linear(in_features=2560, out_features=128, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (out): Linear(in_features=64, out_features=1, bias=True)
)
Total parameters: 336129
Strating training...
Epoch   1/30 | Train Loss: 119.4955 | Test Loss: 41.2816
Epoch  10/30 | Train Loss: 15.0402 | Test Loss: 19.1004
Epoch  20/30 | Train Loss: 15.4521 | Test Loss: 15.3751
Epoch  30/30 | Train Loss: 14.1716 | Test Loss: 14.5588
Training complete.

=== Test Set Metrics ===
MSE: 14.5588
RMSE: 3.8156
R²: 0.8909
MAE: 3.1374
Plot saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\visuals\training_curves.png
Model saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\curing_predictor_model.pth
Model config saved to model_config.json
```
The model generalizes well, and **overfitting was almost completely eliminated** simply by reducing the number of epochs.

Now let's introduce **dropout**.  
It randomly "switches off" certain neurons during training, forcing the network not to rely on a few specific neurons and improving generalization.

<div style="border-left: 4px solid #8250df; background-color: rgba(130, 80, 223, 0.1); padding: 10px 15px; margin: 10px 0;">
    <strong style="color: #8250df;">NEXT FIX:</strong><br> 
    <b>The row <code>self.dropout = nn.Dropout(dropout_rate)</code> in the <code>model.py</code> module</b> is added right after the ReLU row in the model class init, <b>and the droput application row (<code>x = self.dropout(x)</code>)</b> is added after each activation in the forward method.
</div>

## 3rd version output

```text
Loading of PIs embeddings...
Loading of Monomers embeddings...
PIs and Monomers embeddings loaded successfully.
    PIs embeddings: (208, 1280)
    Monomers embeddings: (40, 1280)
Creating the combined dataset...
Combined dataset shape: (8320, 2560)
Generating simulated target values...
Target values shape: (8320,)
Min: 6.33%, Max: 59.88%
Dataset successfully splitted for training and testing.
    Train set: 6656 samples
    Test set: 1664 samples
Train batches: 104
Test batches: 26
CuringPredictorNet(
  (fc1): Linear(in_features=2560, out_features=128, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.3, inplace=False)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (out): Linear(in_features=64, out_features=1, bias=True)
)
Total parameters: 336129
Strating training...
Epoch   1/30 | Train Loss: 128.7929 | Test Loss: 64.2913
Epoch  10/30 | Train Loss: 47.7390 | Test Loss: 16.5619
Epoch  20/30 | Train Loss: 34.9018 | Test Loss: 16.2991
Epoch  30/30 | Train Loss: 31.1524 | Test Loss: 27.7807
Training complete.

=== Test Set Metrics ===
MSE: 27.7807
RMSE: 5.2707
R²: 0.7918
MAE: 4.3445
Plot saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\visuals\training_curves.png
Model saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\curing_predictor_model.pth
Model config saved to model_config.json
```
The model performance significantly dropped.  
This is a perfectly normal phenomenon with synthetic data, but it does not apply to real data.  

The droput role according to the data context:
* Destructive (synthetic data): "switches off" neurons that are essential for capturing the exact mathematical pattern.	
* Protective (real data): prevents the model from learning noise specific to the training batch, forcing it to find more general and robust patterns.

>[IMPORTANT]
>Once we will have the lab data, we will definitely need to consider dropout (or other regularization techniques).  
>We will keep dropout in the code but leave it disabled for now. Instead of removing it, we set `dropout_rate=0.0` in `model.py`.  
>Other regulation techniques to consider:
>   - **L2 Regularization (Weight Decay)**: already available in Adam (`weight_decay=1e-4`). It helps keep        weights small and stable.
>   - **Batch Normalization**: stabilizes learning and can reduce the need for dropout.
>   - **Early Stopping with a Validation Set**: Crucial for real-world data. Synthetic data converges predictably, whereas real-world data does not. Early stopping will save the model from overfitting.

Then we go on with model complexity reduction.  
With 8,320 samples, a model with two hidden layers of 128 and 64 neurons (336k parameters) is likely too large and powerful. Let's **reduce the number of neurons**.

<div style="border-left: 4px solid #8250df; background-color: rgba(130, 80, 223, 0.1); padding: 10px 15px; margin: 10px 0;">
    <strong style="color: #8250df;">NEXT FIX:</strong><br> 
    <b>The model initialization in <code>train_regressor.py</code> is modified  to <code>model = CuringPredictorNet(input_dim, hidden1=64, hidden2=32)</code></b>. <code>hidden1</code> changes from 128 to <b>64</b> and <code>hidden2</code> changes from 64 to <b>32</b>.
</div>

## 4th version output
```text
Loading of PIs embeddings...
Loading of Monomers embeddings...
PIs and Monomers embeddings loaded successfully.
    PIs embeddings: (208, 1280)
    Monomers embeddings: (40, 1280)
Creating the combined dataset...
Combined dataset shape: (8320, 2560)
Generating simulated target values...
Target values shape: (8320,)
Min: 6.33%, Max: 59.88%
Dataset successfully splitted for training and testing.
    Train set: 6656 samples
    Test set: 1664 samples
Train batches: 104
Test batches: 26
CuringPredictorNet(
  (fc1): Linear(in_features=2560, out_features=64, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=64, out_features=32, bias=True)
  (out): Linear(in_features=32, out_features=1, bias=True)
)
Total parameters: 166017
Strating training...
Epoch   1/30 | Train Loss: 85.7699 | Test Loss: 39.9160
Epoch  10/30 | Train Loss: 15.7769 | Test Loss: 18.2021
Epoch  20/30 | Train Loss: 14.6674 | Test Loss: 15.7803
Epoch  30/30 | Train Loss: 13.5586 | Test Loss: 15.1638
Training complete.

=== Test Set Metrics ===
MSE: 15.1638
RMSE: 3.8941
R²: 0.8864
MAE: 3.1734
Plot saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\visuals\training_curves.png
Model saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\curing_predictor_model.pth
Model config saved to model_config.json
```
The model performance is quite comparable to the best one so far, but yet slightly worse.  
### Tests Comparison
| Configuration | Test Loss | R² | RMSE | MAE |
| --- | --- | --- | --- | --- |
| 128/64, 30 epochs (best) | 14.55 | 0.89 | 3.81 | 3.13 |
| 64/32, 30 epochs (current) | 15.16 | 0.88 | 3.89 | 3.17 |

I go back to previous configuration and I proceed with **reducing the learning rate**.

<div style="border-left: 4px solid #8250df; background-color: rgba(130, 80, 223, 0.1); padding: 10px 15px; margin: 10px 0;">
    <strong style="color: #8250df;">NEXT FIX:</strong><br> 
    <b>The <code>optimizer</code> definition in the <code>LOSS & OPTIMIZER</code> section is modified to <code>optimizer = optim.Adam(model.parameters(), lr=0.001)</code></b>. <code>lr=0.01</code> is too high for 100 epochs (too many epochs - model reaches fast the minimum loss but it doesen't really stabilize), but <code>lr=0.001</code> is too low for 30 epochs as the model has at this point (the model doesn't have enough time to reduce loss): <b>we now change the learning Rate to 0.001 but also the <code>NUM_EPOCHS</code> back to 100</b>.
</div>

With this configuration, the model might yield a slightly lower loss (e.g., 14.2) or a similar one, but it should be more stable.

## 5th version output
```text
Loading of PIs embeddings...
Loading of Monomers embeddings...
PIs and Monomers embeddings loaded successfully.
    PIs embeddings: (208, 1280)
    Monomers embeddings: (40, 1280)
Creating the combined dataset...
Combined dataset shape: (8320, 2560)
Generating simulated target values...
Target values shape: (8320,)
Min: 6.33%, Max: 59.88%
Dataset successfully splitted for training and testing.
    Train set: 6656 samples
    Test set: 1664 samples
Train batches: 104
Test batches: 26
CuringPredictorNet(
  (fc1): Linear(in_features=2560, out_features=128, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (out): Linear(in_features=64, out_features=1, bias=True)
)
Total parameters: 336129
Strating training...
Epoch   1/100 | Train Loss: 160.1383 | Test Loss: 69.4596
Epoch  10/100 | Train Loss: 15.7663 | Test Loss: 15.2876
Epoch  20/100 | Train Loss: 13.3984 | Test Loss: 14.2258
Epoch  30/100 | Train Loss: 13.2546 | Test Loss: 14.1235
Epoch  40/100 | Train Loss: 12.3443 | Test Loss: 14.4017
Epoch  50/100 | Train Loss: 11.6314 | Test Loss: 15.4572
Epoch  60/100 | Train Loss: 10.9800 | Test Loss: 15.2308
Epoch  70/100 | Train Loss: 10.3401 | Test Loss: 16.0330
Epoch  80/100 | Train Loss: 9.7637 | Test Loss: 15.6598
Epoch  90/100 | Train Loss: 9.4867 | Test Loss: 16.7782
Epoch 100/100 | Train Loss: 8.7391 | Test Loss: 16.5835
Training complete.

=== Test Set Metrics ===
MSE: 16.5835
RMSE: 4.0723
R²: 0.8757
MAE: 3.2516
Plot saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\visuals\training_curves.png
Model saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\curing_predictor_model.pth
Model config saved to model_config.json
```
The lower learning rate (0.001) allowed the model to reach a deeper minimum during training (Train Loss dropped from 14.17 to 8.73!), but it also learned the noise in that synthetic data. Its ability to generalize worsened (Test Loss rose to 16.58).

Let's compare minimum test loss for this version and the best one so far.
### Tests Comparison on respective Minimum Test Losses
| Configuration | Test Loss | R² | RMSE | MAE |
| --- | --- | --- | --- | --- |
| 30th epoch out of 100, lr=0.001 (current) | 14.12 | 0.87 | 4.07 | 3.25 |
| 30th epoch out of 30, lr=0.01 (best) | 14.55 | 0.89 | 3.81 | 3.13 |

With lr=0.01: the loss drops rapidly, reaches its minimum at ~30 epochs, and stabilizes. Had you continued to 100 epochs (as you did), it would have risen slowly, but in a predictable manner.
With lr=0.001: the loss drops more slowly and is still in a downward phase at 30 epochs. The fact that it stands at 14.12 at 30 epochs is a statistical fluke, not a stable equilibrium point.
Looking at the output for lr=0.001, the value was 14.22 at 20 epochs, 14.12 at 30, and 14.40 at 40. The minimum is unstable, and the loss fluctuates.

I go back to the best version so far, with `NUM_EPOCHS = 30` and `lr = 0.01` and try applying the **Weight Decay (L2 Regularization)**, so it's already there when I will implement it for running the model on real data.
It's a technique that **adds a penalty to the model's weights during training**.   
```math
Total Loss = Original Loss + λ * Σ(w_i²)
```  
Where:  
* $λ$ (lambda) is the weight decay coefficient.
* $Σ(w_i²)$ is the sum of the squares of all the weights.

It prevents the model from placing too much importance on individual features (or combinations of features), forcing it to distribute the "weight" of learning across multiple neurons, thereby making it **more robust and less prone to memorizing noise (overfitting)**.

<div style="border-left: 4px solid #8250df; background-color: rgba(130, 80, 223, 0.1); padding: 10px 15px; margin: 10px 0;">
    <strong style="color: #8250df;">NEXT FIX:</strong><br> 
    <b>The <code>optimizer</code> definition in the <code>LOSS & OPTIMIZER</code> section is modified to <code>optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)</code></b>. To maintain flexibility (as I did with dropout), I add weight decay as a configurable parameter in the <code>CONFIGURATION</code> section (<code>WEIGHT_DECAY = 1e-4</code>, 0.0 for disabling it).
</div>

## 6th version output
```text
Loading of PIs embeddings...
Loading of Monomers embeddings...
PIs and Monomers embeddings loaded successfully.
    PIs embeddings: (208, 1280)
    Monomers embeddings: (40, 1280)
Creating the combined dataset...
Combined dataset shape: (8320, 2560)
Generating simulated target values...
Target values shape: (8320,)
Min: 6.33%, Max: 59.88%
Dataset successfully splitted for training and testing.
    Train set: 6656 samples
    Test set: 1664 samples
Train batches: 104
Test batches: 26
CuringPredictorNet(
  (fc1): Linear(in_features=2560, out_features=128, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (out): Linear(in_features=64, out_features=1, bias=True)
)
Total parameters: 336129
Strating training...
Epoch   1/30 | Train Loss: 74.7739 | Test Loss: 31.4645
Epoch  10/30 | Train Loss: 15.7126 | Test Loss: 15.2192
Epoch  20/30 | Train Loss: 15.2536 | Test Loss: 15.7050
Epoch  30/30 | Train Loss: 13.5565 | Test Loss: 15.5525
Training complete.

=== Test Set Metrics ===
MSE: 15.5525
RMSE: 3.9437
R²: 0.8834
MAE: 3.2261
Plot saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\visuals\training_curves.png
Model saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\curing_predictor_model.pth
Model config saved to model_config.json
```

The training loss continues to decrease (from 15.25 to 13.55), while the test loss fluctuates (15.70 → 15.55). This indicates that the model:
* Has not yet reached its stable minimum (training loss is still dropping).
* Is beginning to slightly overfit (the train-test gap is widening).

**The weight decay is slowing down learning, so the model might need more epochs to converge.**

<div style="border-left: 4px solid #8250df; background-color: rgba(130, 80, 223, 0.1); padding: 10px 15px; margin: 10px 0;">
    <strong style="color: #8250df;">NEXT FIX:</strong><br> 
    <b>Set <code>NUM_EPOCHS = 50</code> in the RUN TRAINING section</b> and run it again. I want to confirm if there is an improving or just worsening of overfiting effect.
</div>

## 7th version output
```text
Loading of PIs embeddings...
Loading of Monomers embeddings...
PIs and Monomers embeddings loaded successfully.
    PIs embeddings: (208, 1280)
    Monomers embeddings: (40, 1280)
Creating the combined dataset...
Combined dataset shape: (8320, 2560)
Generating simulated target values...
Target values shape: (8320,)
Min: 6.33%, Max: 59.88%
Dataset successfully splitted for training and testing.
    Train set: 6656 samples
    Test set: 1664 samples
Train batches: 104
Test batches: 26
CuringPredictorNet(
  (fc1): Linear(in_features=2560, out_features=128, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (out): Linear(in_features=64, out_features=1, bias=True)
)
Total parameters: 336129
Strating training...
Epoch   1/50 | Train Loss: 127.9880 | Test Loss: 41.8952
Epoch  10/50 | Train Loss: 15.7682 | Test Loss: 14.7917
Epoch  20/50 | Train Loss: 13.9539 | Test Loss: 15.4887
Epoch  30/50 | Train Loss: 13.7781 | Test Loss: 14.4250
Epoch  40/50 | Train Loss: 13.0876 | Test Loss: 15.2646
Epoch  50/50 | Train Loss: 13.0181 | Test Loss: 15.2654
Training complete.

=== Test Set Metrics ===
MSE: 15.2654
RMSE: 3.9071
R²: 0.8856
MAE: 3.1905
Plot saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\visuals\training_curves.png
Model saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\curing_predictor_model.pth
Model config saved to model_config.json
```
The outcome is better then the previous one with weight decay and 30 epochs, but still worse than the original one with starting conditions and 30 epochs.

Also I just realized I didn't set a random.seed for PyTorch:  
<div style="border-left: 4px solid #8250df; background-color: rgba(130, 80, 223, 0.1); padding: 10px 15px; margin: 10px 0;">
    <strong style="color: #8250df;">NEXT FIX:</strong><br> 
    <b>Add <code>SEED</code> section with random.seed conditional handling on PyTorch and run it all over again</b> to check actual differencies between outputs.
</div>

---
## 1st BIS version output
```text
Loading of PIs embeddings...
Loading of Monomers embeddings...
PIs and Monomers embeddings loaded successfully.
    PIs embeddings: (208, 1280)
    Monomers embeddings: (40, 1280)
Creating the combined dataset...
Combined dataset shape: (8320, 2560)
Generating simulated target values...
Target values shape: (8320,)
Min: 6.33%, Max: 59.88%
Dataset successfully splitted for training and testing.
    Train set: 6656 samples
    Test set: 1664 samples
Train batches: 104
Test batches: 26
CuringPredictorNet(
  (fc1): Linear(in_features=2560, out_features=128, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (out): Linear(in_features=64, out_features=1, bias=True)
)
Total parameters: 336129
Strating training...
Epoch   1/100 | Train Loss: 76.1657 | Test Loss: 48.5403
Epoch  10/100 | Train Loss: 17.7036 | Test Loss: 15.6360
Epoch  20/100 | Train Loss: 13.5062 | Test Loss: 14.3793
Epoch  30/100 | Train Loss: 15.5869 | Test Loss: 16.5776
Epoch  40/100 | Train Loss: 13.2298 | Test Loss: 31.0904
Epoch  50/100 | Train Loss: 12.1427 | Test Loss: 18.0072
Epoch  60/100 | Train Loss: 10.8801 | Test Loss: 16.1266
Epoch  70/100 | Train Loss: 9.7025 | Test Loss: 16.8899
Epoch  80/100 | Train Loss: 9.4493 | Test Loss: 23.2110
Epoch  90/100 | Train Loss: 9.1992 | Test Loss: 18.7382
Epoch 100/100 | Train Loss: 7.8798 | Test Loss: 17.9648
Training complete.

=== Test Set Metrics ===
MSE: 17.9648
RMSE: 4.2385
R²: 0.8654
MAE: 3.3683
Plot saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\visuals\training_curves.png
Model saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\curing_predictor_model.pth
Model config saved to model_config.json
```
Go on with reducing epochs number to 30 like before.

## 2nd BIS version output
```text
Loading of PIs embeddings...
Loading of Monomers embeddings...
PIs and Monomers embeddings loaded successfully.
    PIs embeddings: (208, 1280)
    Monomers embeddings: (40, 1280)
Creating the combined dataset...
Combined dataset shape: (8320, 2560)
Generating simulated target values...
Target values shape: (8320,)
Min: 6.33%, Max: 59.88%
Dataset successfully splitted for training and testing.
    Train set: 6656 samples
    Test set: 1664 samples
Train batches: 104
Test batches: 26
CuringPredictorNet(
  (fc1): Linear(in_features=2560, out_features=128, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (out): Linear(in_features=64, out_features=1, bias=True)
)
Total parameters: 336129
Strating training...
Epoch   1/30 | Train Loss: 76.1657 | Test Loss: 48.5403
Epoch  10/30 | Train Loss: 17.7036 | Test Loss: 15.6360
Epoch  20/30 | Train Loss: 13.5062 | Test Loss: 14.3793
Epoch  30/30 | Train Loss: 15.5869 | Test Loss: 16.5776
Training complete.

=== Test Set Metrics ===
MSE: 16.5776
RMSE: 4.0716
R²: 0.8758
MAE: 3.3171
Plot saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\visuals\training_curves.png
Model saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\curing_predictor_model.pth
Model config saved to model_config.json
```

## try restoring weight decay
```text
Loading of PIs embeddings...
Loading of Monomers embeddings...
PIs and Monomers embeddings loaded successfully.
    PIs embeddings: (208, 1280)
    Monomers embeddings: (40, 1280)
Creating the combined dataset...
Combined dataset shape: (8320, 2560)
Generating simulated target values...
Target values shape: (8320,)
Min: 6.33%, Max: 59.88%
Dataset successfully splitted for training and testing.
    Train set: 6656 samples
    Test set: 1664 samples
Train batches: 104
Test batches: 26
CuringPredictorNet(
  (fc1): Linear(in_features=2560, out_features=128, bias=True)
  (relu): ReLU()
  (dropout): Dropout(p=0.0, inplace=False)
  (fc2): Linear(in_features=128, out_features=64, bias=True)
  (out): Linear(in_features=64, out_features=1, bias=True)
)
Total parameters: 336129
Strating training...
Epoch   1/30 | Train Loss: 77.9051 | Test Loss: 42.0515
Epoch  10/30 | Train Loss: 17.5477 | Test Loss: 16.2750
Epoch  20/30 | Train Loss: 13.6951 | Test Loss: 14.3077
Epoch  30/30 | Train Loss: 15.7377 | Test Loss: 15.2751
Training complete.

=== Test Set Metrics ===
MSE: 15.2751
RMSE: 3.9083
R²: 0.8855
MAE: 3.1844
Plot saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\visuals\training_curves.png
Model saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase3\curing_predictor_model.pth
Model config saved to model_config.json
```
This is my final configuration. 

Proceeding with Phase 4 of the pipeline.

No need to update `train_regressor.py` with environmental features because they will be better handled in Phase 4 by XGBoost, which is best for tabular data and doesn't fail like neural networks with too small data (like in this case, 8k parameters aprox).
