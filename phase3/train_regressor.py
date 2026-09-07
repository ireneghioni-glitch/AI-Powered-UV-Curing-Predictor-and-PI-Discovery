import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.model_selection import train_test_split

from model import CuringPredictorNet


# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).resolve().parent
# data dir path in this sub-folder
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# folder in which embeddings and metadata files we need to access to 
# are located (in phase2/)
DATA_P2_DIR = BASE_DIR.parent / "phase2" / "data"

# PIs
INPUT_PI_EMBED = DATA_P2_DIR / "embeddings_PIs.npy"
INPUT_PI_META = "embeddings_metadata_PIs.csv"

# monomers
INPUT_MONO_EMBED = "embeddings_monomers.npy"
INPUT_MONO_META = "embeddings_metadata_monomers.csv"


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
print(f"Total parameters: {sum(p.numel() 
                               for p in model.parameters())}")

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
# 3.10
