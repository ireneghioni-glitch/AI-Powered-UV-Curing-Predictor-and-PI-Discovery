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
from sklearn.decomposition import PCA
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import json
import sys
import warnings
import hashlib
from itertools import product


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

SOLVENT_BASED = 0
WATER_BASED = 1


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

# --- Average over augmentations ---
def average_augmentations(embeds, meta):
    '''Collapse 4 augmentations of the same molecule into 1 averaged embedding.'''
    unique_names = meta["name"].unique()
    return np.array([
        embeds[meta["name"] == name].mean(axis=0)
        for name in unique_names
    ])

pi_embeds = average_augmentations(pi_embeds, pi_meta)
mono_embeds = average_augmentations(mono_embeds, mono_meta)
pi_meta = pi_meta.drop_duplicates(subset="name").reset_index(drop=True)
mono_meta = mono_meta.drop_duplicates(subset="name").reset_index(drop=True)

print(f"After averaging: PIs {pi_embeds.shape}, Monomers {mono_embeds.shape}")

# --- PCA reduction ---
# PCA cannot extract more components than min(n_samples, n_features) - 1.
# After averaging augmentations, PIs have 52 samples and monomers 10,
# so we cap the components accordingly.
n_pi_pca = min(128, pi_embeds.shape[0] - 1)      # 51
n_mono_pca = min(128, mono_embeds.shape[0] - 1)  # 9

pca_pi = PCA(n_components=n_pi_pca).fit(pi_embeds)
pca_mono = PCA(n_components=n_mono_pca).fit(mono_embeds)
pi_embeds = pca_pi.transform(pi_embeds)
mono_embeds = pca_mono.transform(mono_embeds)

print(f"After PCA: PIs {pi_embeds.shape}, Monomers {mono_embeds.shape}")
print(f"PCA components: PI {n_pi_pca}, Mono {n_mono_pca}")
print(f"Explained variance: PI {pca_pi.explained_variance_ratio_.sum():.2%}, "
      f"Mono {pca_mono.explained_variance_ratio_.sum():.2%}")

# --- Save PCA models for inference ---
import joblib
joblib.dump(pca_pi, BASE_DIR / "pca_pi.pkl")
joblib.dump(pca_mono, BASE_DIR / "pca_mono.pkl")
print(f"PCA models saved: pca_pi.pkl, pca_mono.pkl")


# ==================== DEFINE THE EXPERIMENTAL GRID ====================
'''
Full Factorial Design: for each (PI, monomer) pair, we systematically enumerate
a grid of environmental conditions - Design of Experiments (DoE).

Grid size: 2 × 5 × 10 × 10 = 1,000 conditions per pair
Pairs: 56 PIs × 10 monomers = 560 (after averaging augmentations)
Total rows: 560 × 1,000 = 560,000
Features: 51 + 9 + 4 = 64 (after PCA reduction)
'''
print("Building the factorial design grid...")

# logp from 0.0 to 6.0 included (step of 1.5)
logp_list = np.arange(0.0, 7.5, 1.5).round(1).tolist()
# from 0.5% to 5.0% included (step of 0.5)
pi_conc_list = np.arange(0.5, 5.5, 0.5).round(1).tolist()
# from 100 to 1000 mJ/cm2 (step of 100)
uv_dose_list = np.arange(100, 1100, 100).tolist()

GRID = {
    "is_aqueous":       [SOLVENT_BASED, WATER_BASED],        # 2
    "logp":             logp_list,                           # 5
    "pi_concentration": pi_conc_list,                        # 10
    "uv_dose":          uv_dose_list,                        # 10
}

# Cartesian product → 1,000 (is_aqueous, logp, pi_conc, uv_dose) tuples
grid_conditions = list(product(
    GRID["is_aqueous"],
    GRID["logp"],
    GRID["pi_concentration"],
    GRID["uv_dose"],
))
n_conditions = len(grid_conditions)
print(f"Grid size: {n_conditions} conditions per pair")

# ---- Derive sizes ----
n_pis = pi_embeds.shape[0]      # 224
n_monos = mono_embeds.shape[0]  # 40

# ---- Build the expanded index vectors ----
n_pairs = n_pis * n_monos                 # 560
n_total = n_pairs * n_conditions          # 560,000

# PI index per row: each PI repeats for (n_monos * n_conditions) rows
pair_pi_idx = np.repeat(np.arange(n_pis), n_monos * n_conditions)

# Monomer index per row: within each PI block, monomer cycles with n_conditions
pair_mono_idx = np.tile(
    np.repeat(np.arange(n_monos), n_conditions),
    n_pis,
)

# Sanity check
assert len(pair_pi_idx) == n_total
assert len(pair_mono_idx) == n_total

# ---- Build the visual feature matrix ----
pi_features = pi_embeds[pair_pi_idx]           # (215040, 128)
mono_features = mono_embeds[pair_mono_idx]     # (215040, 128)
X_visual = np.concatenate([pi_features, mono_features], axis=1)  # (215040, 256)
print(f"Visual embeddings shape: {X_visual.shape}")

# ---- Build the environmental feature matrix ----
# The grid repeats once per pair: [24 conditions] [24 conditions] ...
grid_array = np.array(grid_conditions, dtype=np.float32)   # (24, 4)
X_env = np.tile(grid_array, (n_pairs, 1))                  # (215040, 4)
print(f"Environmental features shape: {X_env.shape}")


# ==================== CONCATENATE FEATURES ====================
'''
Concatenation of Visual and Environmental features.
'''
X = np.concatenate([X_visual, X_env], axis=1).astype(np.float32) # float32 to halve the RAM (582 MB)
print(f"Final feature matrix shape: {X.shape}")  # (560000, 64)


# ==================== SIMULATE TARGET VALUES ====================
'''
Deterministic simulation: the "base reactivity" of a (PI, monomer) pair
is derived from a hash of the two names, so the SAME pair always gets the
SAME base across all its 1,000 rows. This is essential: without it, the
model would not be able to isolate the effect of environmental conditions
from the noise in the base.

The remaining two factors (monomer reactivity, UV dose saturation) and the
hydrophobe-water penalty are applied vectorized for speed.
'''
print("Generating simulated target values...")

# I have X, the feature matrix, with shape (560,000, 260). Each row represents 
# a (PI, monomer) pair under a specific experimental condition. 
# I need to produce y: the conversion for each of those 560,000 rows.

# conversione = base(PI, monomero) × fattore(monomero) × dose(UV) × pi(%PI) × mezzo(logp, ambiente)

# related ratios
MONOMER_FACTORS = {
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

# I need to generate a different number for each pair (because every PI-monomer 
# combination has its own intrinsic reactivity) but the same number must appear 
# in all the 1,000 rows for that specific pair.
#                                      ↓
# Controlled randomness - Each pair has its own seed, and I generate its value 
# from that seed. If the pair is always the same, the seed is always the same 
# and the value is always the same use → `hashlib.md5()`.
# I take the string 'TPO-L|TMPTA', encode it into bytes, calculate the MD5, 
# take the first 8 hex characters and convert them into an integer. 

def _stable_pair_seed(pi_name: str, monomer_name: str) -> int:
    '''Deterministic 32-bit seed derived from the (PI, monomer) pair.'''
    key = f"{pi_name}|{monomer_name}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest()[:8], 16)

# I use the seed to create an isolated RNG. Each pair has its own RNG, 
# independent of the others. A number is generated based on the PI type.

def _compute_pair_base(pi_name: str, monomer_name: str, pi_role: str) -> float:
    '''Deterministic base conversion for a (PI, monomer) pair.'''
    rng = np.random.default_rng(_stable_pair_seed(pi_name, monomer_name))
    if pi_role == "PI_TypeI":
        return float(rng.uniform(75, 95))
    if pi_role == "PI_TypeII":
        return float(rng.uniform(50, 75))
    return float(rng.uniform(20, 50))


# ---- Names and roles per row (using the SAME indices as X) ----
# extract names and roles from metadata
pi_names_per_row = pi_meta["name"].values[pair_pi_idx]
pi_roles_per_row = pi_meta["role"].values[pair_pi_idx]
mono_names_per_row = mono_meta["name"].values[pair_mono_idx]

# There are 56 × 10 = 560 unique pairs. 
# I calculate 560 values, store them in an array and then expand them to 560,000 rows.

# ---- Precompute base per pair (n_pairs unique values) ----
# Double loop, 560 iterations. Each iteration performs an MD5 and an RNG operation.
print(f"Precomputing base reactivity for {n_pairs} (PI, monomer) pairs...")
pair_base = np.zeros(n_pairs, dtype=np.float32)
for i in range(n_pis):
    for j in range(n_monos):
        pair_base[i * n_monos + j] = _compute_pair_base(
            pi_meta["name"].iloc[i],
            mono_meta["name"].iloc[j],
            pi_meta["role"].iloc[i],
        )

# For each row, I need to know which pair it belongs to. 
# I already have the PI and monomer indices. I calculate the pair 
# index using the same formula as before.

# ---- Broadcast to all rows ----
pair_idx_per_row = pair_pi_idx * n_monos + pair_mono_idx
base_per_row = pair_base[pair_idx_per_row]

# Same strategy: I pre-calculate an array containing the factor 
# for each row. I use the MONOMER_FACTORS dictionary, with a fallback 
# value of 0.7 for unknown monomers.
# List comprehension on 560,000 elements, followed by conversion to an 
# array.

# ---- Monomer factor per row ----
factor_per_row = np.array(
    [MONOMER_FACTORS.get(m, 0.7) for m in mono_names_per_row],
    dtype=np.float32,
)

# The environmental variables are in X_env, with shape (560000, 4). 
# I extract them separately to apply different operations to each.

# ---- Extract grid columns for vectorized computation ----
is_aqueous_col = X_env[:, 0]
logp_col       = X_env[:, 1]
pi_conc_col    = X_env[:, 2]
uv_dose_col    = X_env[:, 3]

# ---- Factor 1: UV dose saturation ----
# More dose → more radicals, but saturating (first-order kinetics)
dose_factor = 1.0 - np.exp(-0.01 * uv_dose_col)

# ---- Factor 2: PI concentration (bell curve) ----
# Peak efficiency arbitrary fixed at ~3%. Above, UV screening dominates (Beer-Lambert).
pi_optimal = 3.0
pi_factor = (pi_conc_col / pi_optimal) * np.exp(1.0 - (pi_conc_col / pi_optimal))

# ---- Factor 3: Medium compatibility (polarity matching) ----
# Effective LogP of the medium: solvent = 4.0, water = -1.0
# The further the PI LogP is from the medium LogP, the worse it dissolves,
# the fewer radicals reach the monomer.
medium_logp = np.where(is_aqueous_col == 1, -1.0, 4.0)
polarity_mismatch = np.abs(logp_col - medium_logp)
medium_factor = np.exp(-0.15 * polarity_mismatch)

# ---- Combined conversion (multiplicative model) ----
conversion = (
    base_per_row            # pair-specific intrinsic reactivity
    * factor_per_row        # monomer reactivity
    * dose_factor           # UV energy
    * pi_factor             # PI concentration
    * medium_factor         # medium compatibility
)

y = np.clip(conversion, 0.0, 100.0).astype(np.float32)

print(f"Target values shape: {y.shape}")
print(f"Min: {y.min():.2f}%, Max: {y.max():.2f}%, Mean: {y.mean():.2f}%")


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

# Using best_params from v1 (see phase4-development-log.md):
# max_depth=4, learning_rate=0.05, subsample=1.0,
# colsample_bytree=1.0, reg_lambda=2.0, reg_alpha=0
# GridSearch skipped for speed during iterative development.
best_params = {
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'reg_lambda': 2.0,
    'reg_alpha': 0
}
print(f"Using v1 best_params: {best_params}")


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


'''
The whole following section is commented because it has been used in the 
first version of the XGBoost model.
After model refactoring (PCA implementation on embeddings), this model
and the NN one are not comperable anymore as before.'''
# ==================== COMPARE WITH PHASE 3 ===================
# '''
# Load Phase 3 Deep Learning model and evaluate on the same test set.

# The neural network was trained with 2,560 features (visual embeddings only), 
# whereas now you are recreating the model with 2,564 features (adding the 4 
# environmental ones).

# Solution → Recreate the neural network with the correct input dimension 
#            (2560) and use only the embeddings (the first 2560 columns) to 
#            make predictions.
# '''
# print("=== Comparison with Phase 3 Neural Network ===")

# # Recreate the neural network (same architecture as Phase 3)
# nn_input_dim = 2560 # embeddings only
# nn_model = CuringPredictorNet(nn_input_dim, hidden1=128, hidden2=64, dropout_rate=0.0)

# # 2. Load the saved weights
# nn_model.load_state_dict(torch.load(MODEL_PATH_P3, map_location=torch.device('cpu')))
# nn_model.eval()

# # 3. Extract from X_test only first 2560 columns (the embeddings)
# X_test_embeddings = X_test[:, : 256]   # shape: (1248, 2560)

# # 4. Convert test data to PyTorch tensors
# X_test_t = torch.tensor(X_test_embeddings, dtype=torch.float32)

# # 5. Get neural network predictions
# with torch.no_grad():
#     y_pred_nn = nn_model(X_test_t).numpy().flatten()

# # 6. Evaluate neural network on the same test set
# mse_nn = mean_squared_error(y_test, y_pred_nn)
# rmse_nn = np.sqrt(mse_nn)
# mae_nn = mean_absolute_error(y_test, y_pred_nn)
# r2_nn = r2_score(y_test, y_pred_nn)

# print("Neural Network (Phase 3):")
# print(f"  MSE: {mse_nn:.4f}")
# print(f"  RMSE: {rmse_nn:.4f}")
# print(f"  R²: {r2_nn:.4f}")
# print(f"  MAE: {mae_nn:.4f}")

# print("XGBoost (Phase 4):")
# print(f"  MSE: {mse:.4f}")
# print(f"  RMSE: {rmse:.4f}")
# print(f"  R²: {r2:.4f}")
# print(f"  MAE: {mae:.4f}")

# # Display MSE improvement
# mse_improvement = ((mse_nn - mse) / mse_nn) * 100
# print(f"XGBoost improves MSE by: {mse_improvement:.2f}%")


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

     [DEPRECATED IN NEW MODEL VERSION]
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
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Feature Importance
importance = model.feature_importances_
feature_names = [f"Visual_{i}" for i in range(256)] + ["Is_Aqueous", "LogP", "%PI", "UV_Dose"]

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
# axes[2].scatter(y_test, y_pred_nn, alpha=0.5, color='green')
# axes[2].plot([0, 100], [0, 100], 'r--', label="Perfect")
# axes[2].set_xlabel("True Conversion (%)")
# axes[2].set_ylabel("Predicted Conversion (%)")
# axes[2].set_title(f"Neural Network (R² = {r2_nn:.3f})")
# axes[2].legend()
# axes[2].grid(True)

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