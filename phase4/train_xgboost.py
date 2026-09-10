'''
NOTE: `train_xgboost.py` of Phase 4 and `train_regressor.py` from Pase 3,
currently share a few sections which are applying the same logic.
These sections are repeting themeselves in the modules, for pipeline 
construction sake and my ease in studying and building the whole process.

In the future, these section will call the same functions that will be coming
from an `utils` module.
'''

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import json
import sys
import warnings

from early_stopping_wrapper import EarlyStoppingXGBWrapper


# ==================== CONFIGURATION ====================

# current folder (phase4/) in root 
BASE_DIR = Path(__file__).resolve().parent
# data folder in sub-folder
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
# visuals folder in sub-folder
VISUALS_DIR = BASE_DIR / "visuals"
VISUALS_DIR.mkdir(parents=True, exist_ok=True)
# saved model path in this folder
MODEL_PATH = BASE_DIR / "xgboost_model.json"
# saved model config path in this folder
MODEL_PATH_CONFIG = BASE_DIR / "model_config.json"

# folder in which embeddings and metadata files we need to access to 
# are located (data/ in phase2/)
DATA_P2_DIR = BASE_DIR.parent / "phase2" / "data"
# folder in which Deep Learning model file we need to access to 
# is located (phase3/)
P3_DIR = BASE_DIR.parent / "phase3"
sys.path.append(str(P3_DIR))
# import model from Phase 3
from model import CuringPredictorNet    # type: ignore
MODEL_PATH_P3 = BASE_DIR.parent / "phase3" / "curing_predictor_model.pth"

# PIs
INPUT_PI_EMBED = DATA_P2_DIR / "embeddings_PIs.npy"
INPUT_PI_META = DATA_P2_DIR / "embeddings_metadata_PIs.csv"

# monomers
INPUT_MONO_EMBED = DATA_P2_DIR / "embeddings_monomers.npy"
INPUT_MONO_META = DATA_P2_DIR / "embeddings_metadata_monomers.csv"


# ==================== SEED ====================
SEED = 42  # set as 0 for disabling reproducibility and discover new paths to prediction
if SEED > 0:
    np.random.seed(SEED)
    print(f"[INFO] Fixed seed: {SEED} (reproducible results)")
    # torch not strictly needed for XGBoost, but harmless
    torch.manual_seed(SEED)
    # with CUDA:
    torch.cuda.manual_seed_all(SEED)    # if using GPU
else:
    print(f"[INFO] Seed disabled (active exploration)")


# ==================== CONSTANTS ====================

AQUEOUS = 1
SOLVENT = 0


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
HERE IS THE SAME COMBINATION LOGIC FROM `train_regressor.py`.

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
X_visual = np.concatenate([pi_repeated, mono_tiled], axis=1)   # (224*40, 2560)

print(f"Visual embeddings shape: {X_visual.shape}")            # (8960, 2560)


# ==================== GENERATE ENVIRONMENTAL FEATURES ====================
'''
We need the environmental features for each PI‑monomer pair. 
Since we don't have experimental data, we will simulate them for the MVP. 
In a real project, these would come from experimental design.

Simulation logic:
    - Is_Aqueous: Randomly assign 0 (solvent) or 1 (aqueous) for each pair.
    - LogP: Calculate from the PI SMILES using RDKit (we'll simulate for now).
    - % PI: Random value between 1% and 5%.
    - UV_Dose: Random value between 50 and 500 extmJ/cm2.

IMPORTANT!
When real data will be available, I will replace this with CSV loading.
'''
print("Generating environmental features...")

# all data and features from hypothetical experiments
n_samples = n_pis * n_monos

# Simulate all the environmental features
is_aqueous = np.random.choice([SOLVENT, AQUEOUS], size=n_samples)   # 0: solvent, 1: aqueous
logp = np.random.uniform(0.5, 5.0, size=n_samples)     # hydrophobicity (logP)
pi_concentration = np.random.uniform(1.0, 5.0, size=n_samples)  # % PI
uv_dose = np.random.uniform(50, 500, size=n_samples)    # UV light energy density (mJ/cm²)

# Combine them into feature matrix
X_env = np.column_stack([is_aqueous, logp, pi_concentration, uv_dose])
print(f"Environmental features shape: {X_env.shape}")   # (8960, 4)


# ==================== CONCATENATE FEATURES ====================
'''
Concatenation of Visual and Environmental features.
'''
X = np.concatenate([X_visual, X_env], axis=1)
print(f"Final feature matrix shape: {X.shape}")  # (8960, 2564)


# ==================== SIMULATE TARGET VALUES ====================
'''
HERE IS THE SAME SIMULATION LOGIC FROM `train_regressor.py`.

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
      - Aromatic systems (Styrene): exhibit slower kinetics
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

# Function for simulating conversion based on PI and mononomer
def simulate_conversion(
        pi_role, 
        monomer_name, 
        uv_dose=100, 
        is_aqueous=0, 
        logp=2.5):
    '''
    What does this function do?
    ---------------------------
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

    # we introduce a penalty for hydrophobe-water incompatibility
    '''
    General rule of thumb for water solubility:
        - LogP < 1: Hydrophilic molecule (highly water-soluble).
        - LogP between 1 and 3: Moderately hydrophobic 
          (limited water solubility).
        - LogP > 3: Strongly hydrophobic. Water solubility drops 
          drastically (often below 1 mg/L).
    '''
    # penalty by default
    penalty = 1.0   # no penalty

    if is_aqueous == 1 and 3.0 < logp < 4.0:
        penalty = 0.3   # reduces conversion of 30%
    elif is_aqueous == 1 and logp > 4.0:
        '''LogP > 3 indicates strong hydrophobicity. 
        In aqueous media, such PIs tend to precipitate, drastically 
        reducing radical availability.
        MVP penalty: 0.5 (50% conversion drop) to simulate this effect.
        '''
        penalty = 0.5   # reduces conversion of 50%

    conversion = conversion * penalty

    return min(max(conversion, 0), 100)

# Create target values for all combinations
# PI role and monomer name for each row are mandatory infos

# PI
pi_roles = pi_meta['role'].values
pi_roles_repeated = np.repeat(pi_roles, n_monos, axis=0)

# monomer
monomer_names = mono_meta['name'].values
monomer_names_tiled = np.tile(monomer_names, n_pis)

y = np.array([simulate_conversion(
    role, 
    name, 
    uv_dose=uv_dose[i], 
    is_aqueous=is_aqueous[i], 
    logp=logp[i]) for i, (role, name) in enumerate(zip(pi_roles_repeated, monomer_names_tiled))])

print(f"Target values shape: {y.shape}")
print(f"Min: {y.min():.2f}%, Max: {y.max():.2f}%")


# ==================== TRAIN/VALIDATION/TEST SPLIT ====================
'''
Why Validation Set Implementation only in Phase 4
-------------------------------------------------
In Phase 3 (Neural Network), I used a simple Train/Test split (80/20) without a 
validation set. The number of epochs was fixed manually (e.g., 30), and we monitored 
test loss during training to decide when to stop, which is a form of information 
leakage, as I were indirectly peeking at the test set to make training decisions.

In Phase 4 (XGBoost), I decided to introduce directly a Validation Set (15% of the data) for two 
critical reasons:

1. Early Stopping: XGBoost builds trees sequentially. Without early stopping, 
   we would have to guess the optimal number of trees (n_estimators). With a 
   validation set, we can monitor validation loss and stop training automatically 
   when performance stops improving, preventing overfitting and saving computation.

2. Test Set Integrity: The Test Set (15%) remains completely untouched during 
   training and hyperparameter tuning. It is used only once at the very end to 
   evaluate the final model. This ensures an unbiased estimate of generalization 
   performance.

The split is: Train (70%), Validation (15%), Test (15%).
'''
# split into Train and Temporary (70/30)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, 
    y, 
    test_size=0.3,
    random_state=SEED if SEED > 0 else None
)

# Split Temporary (30) into Validation and Test (15/15)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=SEED if SEED > 0 else None
)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# ==================== HYPERPARAMETER TUNING ====================
print("Hyperparameter tuning with GridSearchCV and early stopping...")

# Hyperparameters grid (simple version for MVP)
'''
The grid will be updated and hyperparameter tuning handled 
differently (pyramide tuning).
'''
param_grid = {
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_lambda': [1.0, 2.0],
    'reg_alpha': [0, 0.1]
}

# Wrapper model with early stopping
base_model = EarlyStoppingXGBWrapper(
    n_estimators=1000,
    early_stopping_rounds=10,
    random_state=SEED if SEED > 0 else None,
    eval_set=[(X_val, y_val)]
)

# GridSearch
grid = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train, eval_set=[(X_val, y_val)])
best_params = grid.best_params_
print(f"Best parameters: {best_params}")


# ==================== TRAIN XGBOOST ====================
print("Training final XGBoost with best hyperparameters...")

final_model = xgb.XGBRegressor(
    n_estimators=1000,
    **best_params,
    early_stopping_rounds=10,                   # Stops if loss doesn't get better after 10 rounds
    random_state=SEED if SEED > 0 else None,
    verbosity=0                                 # suppress training messages
)

final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
model = final_model # for evaluation

print(f"Training complete. Best number of trees: {model.best_iteration}")


# ==================== XGBOOST MODEL EVALUATION ====================
'''
Calculate model metrics.
'''
print("Evaluating the XGBoost model...")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== Test Set Metrics ===")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")


# ==================== COMPARE WITH PHASE 3 ===================
'''
Load Phase 3 Deep Learning model and evaluate on the same test set.

The neural network was trained with 2,560 features (visual embeddings only), 
whereas now you are recreating the model with 2,564 features (adding the 4 
environmental ones).

Solution → Recreate the neural network with the correct input dimension 
           (2560) and use only the embeddings (the first 2560 columns) to 
           make predictions.
'''
print("=== Comparison with Phase 3 Neural Network ===")

# Recreate the neural network (same architecture as Phase 3)
nn_input_dim = 2560 # embeddings only
nn_model = CuringPredictorNet(nn_input_dim, hidden1=128, hidden2=64, dropout_rate=0.0)

# 2. Load the saved weights
nn_model.load_state_dict(torch.load(MODEL_PATH_P3, map_location=torch.device('cpu')))
nn_model.eval()

# 3. Extract from X_test only first 2560 columns (the embeddings)
X_test_embeddings = X_test[:, : 2560]   # shape: (1248, 2560)

# 4. Convert test data to PyTorch tensors
X_test_t = torch.tensor(X_test_embeddings, dtype=torch.float32)

# 5. Get neural network predictions
with torch.no_grad():
    y_pred_nn = nn_model(X_test_t).numpy().flatten()

# 6. Evaluate neural network on the same test set
mse_nn = mean_squared_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mse_nn)
mae_nn = mean_absolute_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print("Neural Network (Phase 3):")
print(f"  MSE: {mse_nn:.4f}")
print(f"  RMSE: {rmse_nn:.4f}")
print(f"  R²: {r2_nn:.4f}")
print(f"  MAE: {mae_nn:.4f}")

print("XGBoost (Phase 4):")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²: {r2:.4f}")
print(f"  MAE: {mae:.4f}")

# Display MSE improvement
mse_improvement = ((mse_nn - mse) / mse_nn) * 100
print(f"XGBoost improves MSE by: {mse_improvement:.2f}%")


# ==================== VISUALISATION ====================
'''
Model Performance and Interpretability
--------------------------------------
This block generates a three‑panel figure that provides both a quantitative
and qualitative assessment of the XGBoost model and its comparison with the
Phase 3 Neural Network.

1. Feature Importance (Top 20):
   - Displays the 20 most influential features in the XGBoost model.
   - The importance is based on the average gain (or weight) of each feature
     across all decision trees.
   - Features are ordered vertically, with the most important at the top.
   - This panel helps identify which molecular embeddings and environmental
     parameters drive the curing conversion prediction.
   - Typical expected top features: some visual embeddings (especially those
     capturing functional groups) and environmental features like LogP or
     Is_Aqueous, due to their strong chemical influence.

2. XGBoost Predictions vs True Values:
   - Scatter plot where each point represents a test sample.
   - X‑axis: true conversion percentage.
   - Y‑axis: predicted conversion percentage by XGBoost.
   - Red dashed line (y = x) indicates perfect prediction.
   - Points close to this line indicate accurate predictions.
   - The R² value is displayed in the title, giving a single‑number summary
     of explained variance.

3. Neural Network Predictions vs True Values:
   - Same scatter plot structure as panel 2, but using predictions from the
     Phase 3 PyTorch neural network.
   - Green points allow easy visual comparison between the two models.
   - This panel serves as a benchmark: since XGBoost is expected to perform
     better on tabular data, its points should cluster more tightly around
     the diagonal line.

Interpretation
--------------
- The three panels together help to:
   * Understand which features matter most (chemistry insight).
   * Visually compare the predictive power of XGBoost vs. the neural network.
   * Identify systematic biases (e.g., over‑prediction at low conversion,
     under‑prediction at high conversion).
   * Validate that XGBoost is the right tool for tabular data
'''
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Feature Importance
importance = model.feature_importances_
feature_names = [f"Visual_{i}" for i in range(2560)] + ["Is_Aqueous", "LogP", "%PI", "UV_Dose"]

# Get top 20 features
top_indices = np.argsort(importance)[-20:]
top_importance = importance[top_indices]
top_names = [feature_names[i] for i in top_indices]

axes[0].barh(top_names, top_importance)
axes[0].set_xlabel("Importance")
axes[0].set_title("Top 20 Feature Importances")

# 2. Predictions vs True (XGBoost)
axes[1].scatter(y_test, y_pred, alpha=0.5)
axes[1].plot([0, 100], [0, 100], 'r--', label="Perfect")
axes[1].set_xlabel("True Conversion (%)")
axes[1].set_ylabel("Predicted Conversion (%)")
axes[1].set_title(f"XGBoost (R² = {r2:.3f})")
axes[1].legend()
axes[1].grid(True)

# 3. Predictions vs True (Neural Network)
axes[2].scatter(y_test, y_pred_nn, alpha=0.5, color='green')
axes[2].plot([0, 100], [0, 100], 'r--', label="Perfect")
axes[2].set_xlabel("True Conversion (%)")
axes[2].set_ylabel("Predicted Conversion (%)")
axes[2].set_title(f"Neural Network (R² = {r2_nn:.3f})")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(VISUALS_DIR / "comparison_plots.png", dpi=100)
print(f"Plot saved to {VISUALS_DIR / 'comparison_plots.png'}")


# ==================== SAVE MODEL ====================
model.save_model(MODEL_PATH)
print(f"XGBoost model saved to {MODEL_PATH}")

# Save model configuration
model_info = {
    "n_estimators": 1000,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "input_dim": X.shape[1],
    "n_samples": X.shape[0]
}
with open(MODEL_PATH_CONFIG, "w") as f:
    json.dump(model_info, f, indent=2)
print("Model config saved to model_config.json")