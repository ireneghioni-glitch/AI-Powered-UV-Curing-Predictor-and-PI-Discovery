# Phase 4: Tabular Integration & Machine Learning (XGBoost) – Detailed Step‑by‑Step Guide

This guide walks you through the fourth phase of your MVP pipeline. By the end of this phase you will have built, trained, and evaluated an XGBoost regressor that takes a combined feature vector (visual embeddings + environmental parameters) as input and predicts the %Curing Conversion. You will also compare its performance with the neural network regressor from Phase 3.

You will learn how to:
* Understand what XGBoost is and why it excels at tabular data.
* Load the embeddings (PI + Monomer) from Phase 2.
* Generate or load the environmental features (`Is_Aqueous`, `LogP`, `PI_Concentration`, `UV_Dose`).
* Concatenate embeddings and environmental features into a single tabular dataset.
* Train an XGBoost regressor.
* Evaluate the model using regression metrics (MSE, RMSE, $R^2$, MAE).
* Compare XGBoost performance with the PyTorch neural network from Phase 3.
* Save the trained XGBoost model for use in Phase 5 (Reflex web app).

---

## 1. Theoretical Foundations: What is XGBoost?

### 1.1 Why XGBoost for Tabular Data?
XGBoost (eXtreme Gradient Boosting) is a gradient boosting algorithm that builds an ensemble of decision trees sequentially. Each new tree corrects the errors of the previous ones. It has become the go‑to algorithm for tabular data due to its:

| Advantage | Explanation |
| :--- | :--- |
| **Performance** | Often wins Kaggle competitions on structured data. |
| **Speed** | Optimised for parallel processing and cache efficiency. |
| **Regularisation** | Built‑in L1/L2 regularisation to prevent overfitting. |
| **Handling of non‑linearities** | Decision trees naturally capture complex, non‑linear relationships without needing manual feature engineering. |
| **Robustness** | Works well with outliers and missing values. |
| **Interpretability** | Feature importance scores tell you which features matter most. |

*Reference: 01-image-classification-theory.md – Section "Image Classification Using Machine Learning"*

### 1.2 How XGBoost Works (In Brief)
XGBoost builds an ensemble of decision trees sequentially:
1. Start with a base prediction (e.g., the mean conversion).
2. Calculate the residuals (errors) of the current model.
3. Train a new decision tree to predict the residuals.
4. Add the tree to the ensemble with a learning rate (shrinkage) to control the step size.
5. Repeat for a specified number of trees (`n_estimators`).

```
Initial prediction → Compute residuals → Train tree on residuals → Add tree (weighted) → Update prediction → Repeat
```

Each tree is a weak learner, but together they form a strong predictor. The gradient part comes from using gradient descent to minimise the loss function.

*Reference: 03-the-deep-in-deep-learning.pdf – Section "Gradient and gradient descent"*

### 1.3 Neural Network vs XGBoost: When to Use Which?

| Aspect | Neural Network (Phase 3) | XGBoost (Phase 4) |
| :--- | :--- | :--- |
| **Data type** | Images, sequences, unstructured data | Tabular data (rows and columns) |
| **Feature engineering** | Automatic (learns features) | Requires manual feature engineering |
| **Interpretability** | Black box (hard to explain) | Feature importance available |
| **Training time** | Slower (needs epochs, GPUs) | Faster (CPU‑optimised) |
| **Data size** | Works well with large datasets | Works well with small‑medium datasets |
| **Handling of non‑linearities** | Excellent (with enough data) | Excellent (tree‑based) |
| **Overfitting** | Requires regularisation (dropout, weight decay) | Built‑in regularisation |

**In your project:** You have tabular data (embeddings + environmental features). XGBoost is a natural choice and will serve as a strong baseline to compare with your neural network.

---

## 2. The Specification (What Phase 4 Requires)

The MVP Technical Specification (`curing-prediction-pipeline-mvp-v3-en.md`) defines Phase 4 as:

**"Tabular Integration & Machine Learning"**:
* Merge visual embeddings with physico‑chemical parameters into a single tabular vector.
* Train an XGBoost regressor on the combined features.
* Predict the continuous target `Double_Bond_Conversion_Percentage` (0–100%).

**The feature vector:**
```
Row = [Embed Monomer] + [Embed PI] + [Is_Aqueous] + [LogP] + [% PI] + [Dose UV]
```

Where:
* **Embed Monomer**: 1280‑dimensional vector from Phase 2.
* **Embed PI**: 1280‑dimensional vector from Phase 2.
* **Is_Aqueous**: Binary flag (1 if water‑based, 0 if solvent‑based).
* **LogP**: Octanol‑water partition coefficient of the PI (calculated via RDKit).
* **% PI**: Weight percentage of photoinitiator in the formulation.
* **Dose UV**: Total radiant energy delivered to the film ($	ext{mJ/cm}^2$).

**Total features:** $1280 + 1280 + 1 + 1 + 1 + 1 = 2564$.

---

## 3. Step‑by‑Step Implementation

### Step 3.1: Environment Setup
Ensure you have the necessary libraries installed. In your `chemvision` environment:

```bash
pip install xgboost scikit-learn numpy pandas matplotlib
```

*Reference: 01-installation.pdf – Section "Installing tensorflow" (adapted for XGBoost)*

### Step 3.2: Project Structure
Create a new folder `phase4/` with the following structure:

```
phase4/
├── data/                       # output files
├── visuals/                    # plots
├── train_xgboost.py            # main script
└── model_config.json           # optional, for saving parameters
```

### Step 3.3: Load Embeddings and Metadata
We will load the embeddings and metadata from Phase 2 for both PIs and monomers.

In `train_xgboost.py`:

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt
import json

# ==================== CONFIGURATION ====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
VISUALS = BASE_DIR / "visuals"
VISUALS.mkdir(parents=True, exist_ok=True)

# Paths to Phase 2 embeddings and metadata
DATA_P2_DIR = BASE_DIR.parent / "phase2" / "data"

# PIs
INPUT_PI_EMBED = DATA_P2_DIR / "embeddings_PIs.npy"
INPUT_PI_META = DATA_P2_DIR / "embeddings_metadata_PIs.csv"

# Monomers
INPUT_MONO_EMBED = DATA_P2_DIR / "embeddings_monomers.npy"
INPUT_MONO_META = DATA_P2_DIR / "embeddings_metadata_monomers.csv"

# ==================== LOAD DATA ====================
print("Loading PI embeddings...")
pi_embeds = np.load(INPUT_PI_EMBED)      # (224, 1280)
pi_meta = pd.read_csv(INPUT_PI_META)

print("Loading Monomer embeddings...")
mono_embeds = np.load(INPUT_MONO_EMBED)  # (40, 1280)
mono_meta = pd.read_csv(INPUT_MONO_META)

print(f"PI embeddings: {pi_embeds.shape}")
print(f"Monomer embeddings: {mono_embeds.shape}")
```

### Step 3.4: Create Combined Dataset (PI + Monomer)
Just like in Phase 3, we create all possible PI‑monomer pairs (Cartesian product).

```python
# ==================== COMBINE DATASETS ====================
print("Creating combined dataset...")
n_pis = pi_embeds.shape[0]      # 224
n_monos = mono_embeds.shape[0]  # 40

# Repeat PI embeddings for each monomer
pi_repeated = np.repeat(pi_embeds, n_monos, axis=0)  # (8960, 1280)

# Tile monomer embeddings for each PI
mono_tiled = np.tile(mono_embeds, (n_pis, 1))        # (8960, 1280)

# Visual embeddings (concatenated)
X_visual = np.concatenate([pi_repeated, mono_tiled], axis=1)  # (8960, 2560)
print(f"Visual embeddings shape: {X_visual.shape}")  # (8960, 2560)
```

### Step 3.5: Generate Environmental Features
We need the environmental features for each PI‑monomer pair. Since we don't have experimental data, we will simulate them for the MVP. In a real project, these would come from your experimental design.

**Simulation logic:**
* `Is_Aqueous`: Randomly assign 0 (solvent) or 1 (aqueous) for each pair.
* `LogP`: Calculate from the PI SMILES using RDKit (we'll simulate for now).
* `% PI`: Random value between 1% and 5%.
* `UV_Dose`: Random value between 50 and 500 $	ext{mJ/cm}^2$.

```python
# ==================== GENERATE ENVIRONMENTAL FEATURES ====================
print("Generating environmental features...")
np.random.seed(42)

n_samples = n_pis * n_monos

# Simulate environmental features
is_aqueous = np.random.choice([0, 1], size=n_samples)  # 0: solvent, 1: aqueous
logp = np.random.uniform(0.5, 5.0, size=n_samples)     # hydrophobicity (logP)
pi_concentration = np.random.uniform(1.0, 5.0, size=n_samples)  # % PI
uv_dose = np.random.uniform(50, 500, size=n_samples)   # mJ/cm²

# Combine into feature matrix
X_env = np.column_stack([is_aqueous, logp, pi_concentration, uv_dose])
print(f"Environmental features shape: {X_env.shape}")  # (8960, 4)
```

*When you have real data: Replace the simulation with actual values loaded from a CSV file.*

### Step 3.6: Concatenate Visual and Environmental Features

```python
# ==================== CONCATENATE FEATURES ====================
X = np.concatenate([X_visual, X_env], axis=1)
print(f"Final feature matrix shape: {X.shape}")  # (8960, 2564)
```

### Step 3.7: Simulate Target Values
We reuse the same simulation logic from Phase 3:

```python
# ==================== SIMULATE TARGET VALUES ====================
print("Generating simulated target values...")
np.random.seed(42)

def monomer_factor():
    monomer_factors_dict = {
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
    return monomer_factors_dict

def simulate_conversion(pi_role, monomer_name, uv_dose=100):
    if pi_role == "PI_TypeI":
        base = np.random.uniform(75, 95)
    elif pi_role == "PI_TypeII":
        base = np.random.uniform(50, 75)
    else:
        base = np.random.uniform(20, 50)
    
    monomer_factors = monomer_factor()
    factor = monomer_factors.get(monomer_name, 0.7)
    
    dose_factor = 1 - np.exp(-0.01 * uv_dose)
    conversion = base * factor * dose_factor
    return min(max(conversion, 0), 100)

# Create target values for all combinations
pi_roles = pi_meta['role'].values
pi_roles_repeated = np.repeat(pi_roles, n_monos, axis=0)

monomer_names = mono_meta['name'].values
monomer_names_tiled = np.tile(monomer_names, n_pis)

y = np.array([simulate_conversion(role, name) for role, name in zip(pi_roles_repeated, monomer_names_tiled)])

print(f"Target values shape: {y.shape}")
print(f"Min: {y.min():.2f}%, Max: {y.max():.2f}%")
```

### Step 3.8: Split Data into Train/Test Sets

```python
# ==================== TRAIN/TEST SPLIT ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

### Step 3.9: Train XGBoost Regressor
XGBoost has many hyperparameters. We'll start with a baseline configuration and then tune.

```python
# ==================== TRAIN XGBOOST ====================
print("Training XGBoost regressor...")

model = xgb.XGBRegressor(
    n_estimators=100,          # number of trees
    learning_rate=0.1,         # step size shrinkage
    max_depth=6,               # maximum tree depth
    subsample=0.8,             # fraction of samples used per tree
    colsample_bytree=0.8,      # fraction of features used per tree
    reg_alpha=0.1,             # L1 regularisation
    reg_lambda=1.0,            # L2 regularisation
    random_state=42,
    verbosity=0                # suppress training messages
)

model.fit(X_train, y_train)
print("Training complete.")
```

**Parameter explanations:**

| Parameter | Meaning | Typical range |
| :--- | :--- | :--- |
| `n_estimators` | Number of trees | 100–1000 |
| `learning_rate` | Shrinkage (step size) | 0.01–0.3 |
| `max_depth` | Tree depth | 3–10 |
| `subsample` | Fraction of data per tree | 0.5–1.0 |
| `colsample_bytree` | Fraction of features per tree | 0.5–1.0 |
| `reg_alpha` | L1 regularisation | 0–10 |
| `reg_lambda` | L2 regularisation | 0–10 |

#### The process to decide HOW to tune the hyperparameters
##### 1. The Three Hyperparameter "Groups" (The Logic)

Before making changes, understand what you are adjusting.

| Group | Parameters | Main Effect |
| :--- | :--- | :--- |
| **Tree Complexity** | `max_depth` | How "smart" each individual tree is. |
| **Regularization** | `reg_alpha` (L1), `reg_lambda` (L2) | How much you "penalize" complexity to prevent overfitting. |
| **Stochasticity/Robustness** | `subsample`, `colsample_bytree` | How much randomness you introduce during training to prevent overfitting. |
| **Learning Dynamics** | `n_estimators`, `learning_rate` | How many trees are used and the speed of learning. |

---

##### 2. The Tuning Order (The Winning Strategy)

**Don't tune them all at once!** Follow this logical sequence (used by Kaggle data scientists).

* **Step 0**: Set `n_estimators` HIGH (e.g., 1000) and enable Early Stopping

This eliminates the need to guess the number of trees. The model will stop automatically when the validation loss stops decreasing.

* **Step 1**: Set `max_depth` ("Brute Force")

**What it is**: Tree depth. Deep trees learn very specific rules.

**How ​​to decide**:
- If you have few samples or many features (like your 8k samples / 2564 features), use low values ​​(`3` or `4`). Deep trees (`> 6`) will learn the noise in your embeddings.
- Start with `max_depth=6` (the default). If you see overfitting (very low training loss, high validation loss), reduce it to `4` or `3`. If you see underfitting (both losses are high), increase it to `8` or `10`.

**Your case**: With 2564 features but only 8k samples, I would start with `max_depth=4` (to be conservative).

* **Step 2**: Set `subsample` and `colsample_bytree` (The "Robustness" Factors)

**What they do**: They use only a fraction of the data (`subsample`) and features (`colsample_bytree`) to build each tree. This introduces randomness and reduces overfitting.

**How ​​to decide**:
- Values ​​between `0.6` and `0.9` are standard.
- If your dataset is small (like yours), you can keep `subsample` at `1.0` (using all the data) because you have few samples.
- Regarding `colsample_bytree`, your data contains many redundant features (1280 PI embeddings + 1280 monomer embeddings). Using `0.7` or `0.8` forces XGBoost to look at different combinations of features, making it more robust.

**Your case**: Leave `subsample=0.8` and `colsample_bytree=0.8`. It is a good compromise.

* **Step 3**: Set `reg_alpha` and `reg_lambda` (The "Complexity Penalty")

**What they do**: They add a mathematical penalty to the tree weights (leaves).
- `reg_lambda` (L2, default 1.0): "Flattens" the weights. This is the most important one.
- `reg_alpha` (L1, default 0): Sets useless weights to zero (performs feature selection).

**How ​​to decide**:
- If the model overfits (large train-val gap), **increase them** (e.g., `reg_lambda=3.0`, `reg_alpha=1.0`).
- If it underfits (doesn't learn enough), **reduce them** (e.g., `reg_lambda=0.5`, `reg_alpha=0`).

**Your case**: The embeddings are a mix of noise and signal. Setting `reg_lambda=1.5` or `2.0` can greatly help stabilize the model.

* **Step 4**: Adjust `learning_rate` (The "Step Size")

- `learning_rate` and `n_estimators` are inversely proportional.
- `lr=0.3` + `n_estimators=50` ≈ `lr=0.03` + `n_estimators=500`.
- **Rule of thumb**: Since you are using early stopping, you can keep `lr=0.1` (the default) or lower it to `0.05`. Lower learning rates require more trees but often find better minima. For the MVP, `0.1` works perfectly well.

### Step 3.10: Evaluate the Model

```python
# ==================== EVALUATION ====================
print("Evaluating model...")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("
=== Test Set Metrics ===")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")
```

### Step 3.11: Compare with Phase 3 Neural Network
Now, load the Phase 3 model and evaluate it on the same test set for a fair comparison.

```python
# ==================== COMPARE WITH PHASE 3 ====================
print("
=== Comparison with Phase 3 Neural Network ===")

# Load Phase 3 model
import torch
from model import CuringPredictorNet  # you'll need to copy or link this

# Recreate the neural network (same architecture as Phase 3)
input_dim = X_train.shape[1]  # 2564
nn_model = CuringPredictorNet(input_dim, hidden1=128, hidden2=64, dropout_rate=0.0)

# Load the saved weights
MODEL_PATH_P3 = BASE_DIR.parent / "phase3" / "curing_predictor_model.pth"
nn_model.load_state_dict(torch.load(MODEL_PATH_P3, map_location=torch.device('cpu')))
nn_model.eval()

# Convert test data to PyTorch tensors
X_test_t = torch.tensor(X_test, dtype=torch.float32)

# Get neural network predictions
with torch.no_grad():
    y_pred_nn = nn_model(X_test_t).numpy().flatten()

# Evaluate neural network on the same test set
mse_nn = mean_squared_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mse_nn)
mae_nn = mean_absolute_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print("Neural Network (Phase 3):")
print(f"  MSE: {mse_nn:.4f}")
print(f"  RMSE: {rmse_nn:.4f}")
print(f"  R²: {r2_nn:.4f}")
print(f"  MAE: {mae_nn:.4f}")

print("
XGBoost (Phase 4):")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²: {r2:.4f}")
print(f"  MAE: {mae:.4f}")

# Improvement
mse_improvement = ((mse_nn - mse) / mse_nn) * 100
print(f"
XGBoost improves MSE by: {mse_improvement:.2f}%")
```

### Step 3.12: Visualise Results

```python
# ==================== VISUALISATION ====================
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
plt.savefig(VISUALS / "comparison_plots.png", dpi=100)
print(f"Plot saved to {VISUALS / 'comparison_plots.png'}")
```

### Step 3.13: Save the XGBoost Model

```python
# ==================== SAVE MODEL ====================
MODEL_PATH = BASE_DIR / "xgboost_model.json"
model.save_model(MODEL_PATH)
print(f"XGBoost model saved to {MODEL_PATH}")

# Save model configuration
model_info = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "input_dim": X.shape[1],
    "n_samples": X.shape[0]
}
with open(BASE_DIR / "model_config.json", "w") as f:
    json.dump(model_info, f, indent=2)
print("Model config saved to model_config.json")
```

---

## 4. Complete Script Code

Save the script below as `phase4/train_xgboost.py`:

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import matplotlib.pyplot as plt
import json
import torch
import sys

# Add phase3 to path for model import
sys.path.append(str(Path(__file__).resolve().parent.parent / "phase3"))
from model import CuringPredictorNet

# ==================== CONFIGURATION ====================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
VISUALS = BASE_DIR / "visuals"
VISUALS.mkdir(parents=True, exist_ok=True)

DATA_P2_DIR = BASE_DIR.parent / "phase2" / "data"

INPUT_PI_EMBED = DATA_P2_DIR / "embeddings_PIs.npy"
INPUT_PI_META = DATA_P2_DIR / "embeddings_metadata_PIs.csv"
INPUT_MONO_EMBED = DATA_P2_DIR / "embeddings_monomers.npy"
INPUT_MONO_META = DATA_P2_DIR / "embeddings_metadata_monomers.csv"

# ==================== LOAD DATA ====================
print("Loading PI embeddings...")
pi_embeds = np.load(INPUT_PI_EMBED)
pi_meta = pd.read_csv(INPUT_PI_META)

print("Loading Monomer embeddings...")
mono_embeds = np.load(INPUT_MONO_EMBED)
mono_meta = pd.read_csv(INPUT_MONO_META)

print(f"PI embeddings: {pi_embeds.shape}")
print(f"Monomer embeddings: {mono_embeds.shape}")

# ==================== COMBINE DATASETS ====================
print("Creating combined dataset...")
n_pis = pi_embeds.shape[0]
n_monos = mono_embeds.shape[0]

pi_repeated = np.repeat(pi_embeds, n_monos, axis=0)
mono_tiled = np.tile(mono_embeds, (n_pis, 1))
X_visual = np.concatenate([pi_repeated, mono_tiled], axis=1)
print(f"Visual embeddings shape: {X_visual.shape}")

# ==================== GENERATE ENVIRONMENTAL FEATURES ====================
print("Generating environmental features...")
np.random.seed(42)
n_samples = n_pis * n_monos

is_aqueous = np.random.choice([0, 1], size=n_samples)
logp = np.random.uniform(0.5, 5.0, size=n_samples)
pi_concentration = np.random.uniform(1.0, 5.0, size=n_samples)
uv_dose = np.random.uniform(50, 500, size=n_samples)

X_env = np.column_stack([is_aqueous, logp, pi_concentration, uv_dose])
print(f"Environmental features shape: {X_env.shape}")

# ==================== CONCATENATE FEATURES ====================
X = np.concatenate([X_visual, X_env], axis=1)
print(f"Final feature matrix shape: {X.shape}")

# ==================== SIMULATE TARGET VALUES ====================
print("Generating simulated target values...")
np.random.seed(42)

def monomer_factor():
    return {
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

def simulate_conversion(pi_role, monomer_name, uv_dose=100):
    if pi_role == "PI_TypeI":
        base = np.random.uniform(75, 95)
    elif pi_role == "PI_TypeII":
        base = np.random.uniform(50, 75)
    else:
        base = np.random.uniform(20, 50)
    
    monomer_factors = monomer_factor()
    factor = monomer_factors.get(monomer_name, 0.7)
    dose_factor = 1 - np.exp(-0.01 * uv_dose)
    conversion = base * factor * dose_factor
    return min(max(conversion, 0), 100)

pi_roles = pi_meta['role'].values
pi_roles_repeated = np.repeat(pi_roles, n_monos, axis=0)
monomer_names = mono_meta['name'].values
monomer_names_tiled = np.tile(monomer_names, n_pis)

y = np.array([simulate_conversion(role, name) for role, name in zip(pi_roles_repeated, monomer_names_tiled)])
print(f"Target values shape: {y.shape}")
print(f"Min: {y.min():.2f}%, Max: {y.max():.2f}%")

# ==================== TRAIN/TEST SPLIT ====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ==================== TRAIN XGBOOST ====================
print("Training XGBoost regressor...")
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train)
print("Training complete.")

# ==================== EVALUATE XGBOOST ====================
print("Evaluating XGBoost...")
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("
=== XGBoost Test Set Metrics ===")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")

# ==================== COMPARE WITH NEURAL NETWORK ====================
print("
=== Comparing with Phase 3 Neural Network ===")

# Load neural network
input_dim = X_train.shape[1]
nn_model = CuringPredictorNet(input_dim, hidden1=128, hidden2=64, dropout_rate=0.0)

MODEL_PATH_P3 = BASE_DIR.parent / "phase3" / "curing_predictor_model.pth"
nn_model.load_state_dict(torch.load(MODEL_PATH_P3, map_location=torch.device('cpu')))
nn_model.eval()

X_test_t = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    y_pred_nn = nn_model(X_test_t).numpy().flatten()

mse_nn = mean_squared_error(y_test, y_pred_nn)
rmse_nn = np.sqrt(mse_nn)
mae_nn = mean_absolute_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

print("Neural Network (Phase 3):")
print(f"  MSE: {mse_nn:.4f}")
print(f"  RMSE: {rmse_nn:.4f}")
print(f"  R²: {r2_nn:.4f}")
print(f"  MAE: {mae_nn:.4f}")

print("
XGBoost (Phase 4):")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  R²: {r2:.4f}")
print(f"  MAE: {mae:.4f}")

# ==================== VISUALISATION ====================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Feature importance
importance = model.feature_importances_
feature_names = [f"Visual_{i}" for i in range(2560)] + ["Is_Aqueous", "LogP", "%PI", "UV_Dose"]
top_indices = np.argsort(importance)[-20:]
top_importance = importance[top_indices]
top_names = [feature_names[i] for i in top_indices]

axes[0].barh(top_names, top_importance)
axes[0].set_xlabel("Importance")
axes[0].set_title("Top 20 Feature Importances")

# XGBoost predictions
axes[1].scatter(y_test, y_pred, alpha=0.5)
axes[1].plot([0, 100], [0, 100], 'r--', label="Perfect")
axes[1].set_xlabel("True Conversion (%)")
axes[1].set_ylabel("Predicted Conversion (%)")
axes[1].set_title(f"XGBoost (R² = {r2:.3f})")
axes[1].legend()
axes[1].grid(True)

# Neural Network predictions
axes[2].scatter(y_test, y_pred_nn, alpha=0.5, color='green')
axes[2].plot([0, 100], [0, 100], 'r--', label="Perfect")
axes[2].set_xlabel("True Conversion (%)")
axes[2].set_ylabel("Predicted Conversion (%)")
axes[2].set_title(f"Neural Network (R² = {r2_nn:.3f})")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig(VISUALS / "comparison_plots.png", dpi=100)
print(f"Plot saved to {VISUALS / 'comparison_plots.png'}")

# ==================== SAVE MODEL ====================
MODEL_PATH = BASE_DIR / "xgboost_model.json"
model.save_model(MODEL_PATH)
print(f"XGBoost model saved to {MODEL_PATH}")

model_info = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "input_dim": X.shape[1],
    "n_samples": X.shape[0]
}
with open(BASE_DIR / "model_config.json", "w") as f:
    json.dump(model_info, f, indent=2)
print("Model config saved to model_config.json")
```

---

## 5. Hyperparameter Tuning (Optional)

XGBoost has many hyperparameters. You can use `GridSearchCV` or `RandomizedSearchCV` for tuning.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0.5, 1.0, 2.0]
}

grid = GridSearchCV(
    xgb.XGBRegressor(random_state=42, verbosity=0),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print(f"Best parameters: {grid.best_params_}")
print(f"Best CV MSE: {-grid.best_score_:.4f}")

# Use best model
model = grid.best_estimator_
```

---

## 6. Testing and Validation

### 6.1 Verify Model Loading

```python
# Load the saved model
loaded_model = xgb.XGBRegressor()
loaded_model.load_model('phase4/xgboost_model.json')

# Test on a single sample
sample = X_test[0].reshape(1, -1)
prediction = loaded_model.predict(sample)[0]
print(f"Prediction: {prediction:.2f}%")
print(f"True value: {y_test[0]:.2f}%")
```

### 6.2 Interpreting the Results

| Metric | XGBoost | Neural Network | Interpretation |
| :--- | :--- | :--- | :--- |
| **MSE** | Lower is better | Lower is better | XGBoost likely outperforms NN on tabular data |
| **R²** | Higher is better | Higher is better | Closer to 1 = more variance explained |
| **RMSE** | Lower is better | Lower is better | Typical error in percentage points |
| **MAE** | Lower is better | Lower is better | Average absolute error |

---

## 7. Summary of Phase 4 Deliverables

After completing this phase, your project folder should contain:

| File | Description |
| :--- | :--- |
| `phase4/xgboost_model.json` | Trained XGBoost model (can be loaded with `xgb.XGBRegressor().load_model()`). |
| `phase4/model_config.json` | Model configuration (hyperparameters, input dimension). |
| `phase4/visuals/comparison_plots.png` | Visualisation of feature importance and predictions vs true values for both models. |

You have also learned:
* What XGBoost is and why it excels at tabular data.
* How to integrate visual embeddings with environmental features.
* How to train and evaluate an XGBoost regressor.
* How to compare its performance with a neural network.

---

## 8. Next Steps (Preview of Phase 5)

Now that we have both models (neural network and XGBoost), the final phase will:
1. Load the XGBoost model (and optionally the neural network).
2. Build a Reflex web application where users can:
   * Input SMILES strings for PI and monomer.
   * Select environment (aqueous/solvent).
   * Set UV dose and PI concentration.
3. Run the full pipeline on the user's input:
   * Generate molecular images (RDKit).
   * Extract embeddings (MobileNetV2).
   * Concatenate with environmental features.
   * Predict conversion using XGBoost (or neural network).
4. Save predictions to a database (SQLite).

This is where the "Integration & ML" of the pipeline comes to life in a user‑friendly interface.

*Reference: curing-prediction-pipeline-mvp-v3-en.md – Phase 5: “Interactive Deployment with Reflex”*

---

## 9. Theoretical Summary of Key Concepts

### 9.1 Gradient Boosting
Gradient boosting builds an ensemble of weak learners (decision trees) sequentially. Each new tree focuses on the errors (residuals) of the previous trees.

### 9.2 XGBoost Advantages
* **Regularization**: Prevents overfitting (L1/L2).
* **Handling missing values**: Automatically learns the best direction for missing values.
* **Tree pruning**: Stops splitting when no gain is found.
* **Parallel processing**: Fast training on CPU.

### 9.3 Feature Engineering
Combining visual embeddings (from CNNs) with hand‑crafted features (environmental parameters) creates a rich representation that captures both structural and physico‑chemical information.

---

## 10. Conclusion

`train_xgboost.py` successfully completes Phase 4 of the MVP pipeline, training an XGBoost regressor on the combined feature vector (visual embeddings + environmental features). The script:

* ✅ Implements the specification exactly as described.
* ✅ Loads embeddings from Phase 2.
* ✅ Generates environmental features (simulated for MVP).
* ✅ Concatenates features into a single tabular dataset.
* ✅ Trains an XGBoost regressor.
* ✅ Evaluates the model on a test set.
* ✅ Compares performance with the Phase 3 neural network.
* ✅ Saves the trained model for use in Phase 5.

Current status: **Ready for Phase 5 (Reactive Web App with Reflex).**
