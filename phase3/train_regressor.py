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


# 3.5 paragraph
# ==================== TRAIN/TEST SPLIT ====================