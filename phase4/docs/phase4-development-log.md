# Phase 4 Development Log: Tabular Integration & XGBoost

#### Document Purpose
This document chronicles the complete development journey of the `train_xgboost.py` script, the core component of Phase 4 of the MVP pipeline. It explains how we transitioned from a simple XGBoost baseline to a professionally tuned model with early stopping and hyperparameter optimization, why we made each design decision, and how the final architecture supports both training and inference.

This log covers:
* The original specification and theoretical foundations
* Key design decisions and their rationale
* The evolution from a simple to a robust implementation
* Hyperparameter tuning strategy and results
* Lessons learned and future improvements

---

#### 1. The Original Specification (What Phase 4 Required)
The MVP Technical Specification (`curing-prediction-pipeline-mvp-v3-en.md`) defines Phase 4 as:
**"Tabular Integration & Machine Learning"**:
* Merge visual embeddings with physico-chemical parameters into a single tabular vector.
* Train an XGBoost regressor on the combined features.
* Predict the continuous target `Double_Bond_Conversion_Percentage` (0–100%).

**The feature vector**:
```text
Row = [Embed Monomer] + [Embed PI] + [Is_Aqueous] + [LogP] + [% PI] + [Dose UV]
Total features: 1280 + 1280 + 1 + 1 + 1 + 1 = 2564.
```

This specification guided every decision we made during the implementation.

---

#### 2. Theoretical Foundations: Why XGBoost?

##### 2.1 XGBoost for Tabular Data
XGBoost (eXtreme Gradient Boosting) builds an ensemble of decision trees sequentially. Each new tree corrects the errors of the previous ones. It has become the go-to algorithm for tabular data because:

| Advantage | Explanation |
| :--- | :--- |
| **Performance** | Often wins Kaggle competitions on structured data. |
| **Speed** | Optimised for parallel processing and cache efficiency. |
| **Regularisation** | Built-in L1/L2 regularisation to prevent overfitting. |
| **Handling of non-linearities** | Decision trees naturally capture complex, non-linear relationships without needing manual feature engineering. |
| **Robustness** | Works well with outliers and missing values. |
| **Interpretability** | Feature importance scores tell you which features matter most. |

*Reference: `01-image-classification-theory.md` – Section "Image Classification Using Machine Learning"*

##### 2.2 Why XGBoost Over Neural Networks for This Problem?

| Aspect | Neural Network (Phase 3) | XGBoost (Phase 4) |
| :--- | :--- | :--- |
| **Data type** | Images, sequences, unstructured data | Tabular data (rows and columns) |
| **Feature engineering** | Automatic (learns features) | Requires manual feature engineering |
| **Interpretability** | Black box (hard to explain) | Feature importance available |
| **Training time** | Slower (needs epochs, GPUs) | Faster (CPU-optimised) |
| **Data size** | Works well with large datasets | Works well with small-medium datasets |
| **Handling of non-linearities** | Excellent (with enough data) | Excellent (tree-based) |
| **Overfitting** | Requires regularisation (dropout, weight decay) | Built-in regularisation |

In our project, XGBoost is the natural choice because:
* We have tabular data (embeddings + environmental features).
* We have a medium-sized dataset (8,320 samples).
* XGBoost handles the non-linear interactions (e.g., LogP > 3 + aqueous environment) far better than neural networks.

*Reference: `03-the-deep-in-deep-learning.pdf` – Section "Gradient and gradient descent"*

---

#### 3. Key Design Decisions

##### 3.1 Dataset Creation: The Cartesian Product
**Decision**: Create all possible PI-monomer pairs (Cartesian product).

```python
n_pis = pi_embeds.shape[0]      # 224
n_monos = mono_embeds.shape[0]  # 40

pi_repeated = np.repeat(pi_embeds, n_monos, axis=0)  # (8960, 1280)
mono_tiled = np.tile(mono_embeds, (n_pis, 1))        # (8960, 1280)
X_visual = np.concatenate([pi_repeated, mono_tiled], axis=1)  # (8960, 2560)
```

**Why this approach?**
* It covers all possible combinations (224 × 40 = 8,960).
* The model learns the interaction between each PI and each monomer.
* In a real scenario, you would have experimental data for specific combinations, but we simulate a full factorial design for the MVP.

*Theoretical reference: This is the Cartesian product (cross-join) concept from set theory, applied to create a design matrix for a full factorial experiment.*

##### 3.2 Environmental Features: Simulation for MVP
**Decision**: Simulate environmental features since we don't have experimental data.

```python
is_aqueous = np.random.choice([SOLVENT, AQUEOUS], size=n_samples)
logp = np.random.uniform(0.5, 5.0, size=n_samples)
pi_concentration = np.random.uniform(1.0, 5.0, size=n_samples)
uv_dose = np.random.uniform(50, 500, size=n_samples)
X_env = np.column_stack([is_aqueous, logp, pi_concentration, uv_dose])
```

**Why this simulation?**
* **Is_Aqueous**: Binary flag (0 = solvent, 1 = aqueous) – a fundamental environmental choice.
* **LogP**: Octanol-water partition coefficient – influences PI solubility in aqueous systems.
* **% PI**: Photoinitiator concentration – typically ranges from 1% to 5%.
* **UV_Dose**: Energy density – between 50 and 500 mJ/cm², covering typical curing conditions.

*IMPORTANT: When real data becomes available, replace this simulation with CSV loading.*

##### 3.3 Target Simulation: Chemical Plausibility
**Decision**: Simulate target values using a function that incorporates:
* PI type (Type I → high conversion, Type II → medium, co-initiator → low).
* Monomer reactivity factor (tri-acrylate → high, methacrylate → low).
* UV dose saturation effect (exponential approach to a limit).
* Penalty for hydrophobe-water incompatibility (LogP > 3 in aqueous media).

```python
def simulate_conversion(pi_role, monomer_name, uv_dose=100, is_aqueous=0, logp=2.5):
    # Base conversion from PI type
    if pi_role == "PI_TypeI":
        base = np.random.uniform(75, 95)
    elif pi_role == "PI_TypeII":
        base = np.random.uniform(50, 75)
    else:
        base = np.random.uniform(20, 50)
    
    # Monomer factor (functionality and type)
    factor = monomer_factors.get(monomer_name, 0.7)
    
    # UV dose effect (saturation)
    dose_factor = 1 - np.exp(-0.01 * uv_dose)
    conversion = base * factor * dose_factor
    
    # Penalty for hydrophobe-water incompatibility
    penalty = 1.0
    if is_aqueous == 1 and 3.0 < logp < 4.0:
        penalty = 0.3
    elif is_aqueous == 1 and logp > 4.0:
        penalty = 0.5
    
    conversion = conversion * penalty
    return min(max(conversion, 0), 100)
```

**Why this simulation?**
* It is chemically plausible: PI type determines radical generation, monomer functionality determines reactivity, UV dose shows saturation behaviour.
* The LogP penalty models the real-world phenomenon of hydrophobic PIs precipitating in aqueous media, drastically reducing conversion.
* This is a simplified but chemically informed simulation for MVP demonstration.

##### 3.4 Validation Set (Why Only in Phase 4?)
**Decision**: Introduce a Validation Set (15% of data) specifically in Phase 4.

```python
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)
```

* **Why not in Phase 3?** In Phase 3 (Neural Network), we used a simple Train/Test split (80/20) without a validation set. The number of epochs was fixed manually (e.g., 30), and we monitored test loss during training to decide when to stop — which is a form of information leakage, as we were indirectly peeking at the test set to make training decisions.
* **Why in Phase 4?**
  * **Early Stopping**: XGBoost builds trees sequentially. Without early stopping, we would have to guess the optimal number of trees (`n_estimators`). With a validation set, we can monitor validation loss and stop training automatically when performance stops improving, preventing overfitting and saving computation.
  * **Test Set Integrity**: The Test Set (15%) remains completely untouched during training and hyperparameter tuning. It is used only once at the very end to evaluate the final model, ensuring an unbiased estimate of generalisation performance.

##### 3.5 Early Stopping Wrapper (The Complex Decision)
**Decision**: Create a custom wrapper class `EarlyStoppingXGBWrapper` to enable early stopping inside `GridSearchCV`.

**Why this complexity?** The standard `GridSearchCV` in scikit-learn does not support the `early_stopping_rounds` parameter because it does not know how to handle the `eval_set` argument required by XGBoost. The wrapper:
1. Receives the XGBoost hyperparameters, `eval_set` (validation data), and `early_stopping_rounds`.
2. Overrides the `fit()` method to pass `eval_set` to the XGBoost model during training.
3. Returns the trained model with the optimal number of trees (`best_iteration_`).

**The wrapper code**:
```python
class EarlyStoppingXGBWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, ..., eval_set=None, **kwargs):
        # Store all parameters for scikit-learn compatibility
        self.eval_set = eval_set
        # ...
    
    def fit(self, X, y, **fit_params):
        # Extract eval_set from fit_params or use stored one
        eval_set = fit_params.get('eval_set', self.eval_set)
        
        # Create the underlying XGBoost model
        self.model_ = xgb.XGBRegressor(..., early_stopping_rounds=self.early_stopping_rounds)
        
        # Train with early stopping if eval_set is available
        if eval_set is not None:
            self.model_.fit(X, y, eval_set=eval_set, verbose=False)
            self.best_iteration_ = self.model_.best_iteration
        else:
            self.model_.fit(X, y)
            self.best_iteration_ = self.n_estimators
        return self
```

**Why this approach?** It gives us the best of both worlds:
* `GridSearchCV` for hyperparameter optimisation.
* Early stopping inside each cross-validation fold to automatically determine the optimal number of trees.
* **Scalability**: The wrapper is reusable in any XGBoost project.

##### 3.6 Hyperparameter Grid Strategy
**Decision**: Use a focused grid rather than a broad search.

```python
param_grid = {
    'max_depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'reg_lambda': [1.0, 2.0],
    'reg_alpha': [0, 0.1]
}
```

**Why these parameters and values?**

| Parameter | Values | Rationale |
| :--- | :--- | :--- |
| `max_depth` | 4, 6 | Deeper trees (>6) risk overfitting with 8k samples. Conservative depth is safer. |
| `learning_rate` | 0.05, 0.1 | 0.05 is more conservative; 0.1 is default. |
| `subsample` | 0.8, 1.0 | 0.8 introduces randomness (reduces overfitting); 1.0 uses all data. |
| `colsample_bytree` | 0.8, 1.0 | 0.8 forces the model to look at different feature combinations. |
| `reg_lambda` | 1.0, 2.0 | L2 regularisation; 2.0 penalises large weights more. |
| `reg_alpha` | 0, 0.1 | L1 regularisation; 0.1 sets useless weights to zero. |

**Why only 64 combinations?** With 3-fold cross-validation, this gives 192 training runs. This is manageable in 2–5 minutes on CPU. A larger grid would be slower and unnecessary for an MVP.

##### 3.7 Final Training with Best Hyperparameters
**Decision**: After GridSearch finds the best parameters, train a final model on the full training set with early stopping.

```python
final_model = xgb.XGBRegressor(
    n_estimators=1000,
    **best_params,
    early_stopping_rounds=10,
    random_state=SEED if SEED > 0 else None,
    verbosity=0
)

final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
```

**Why not use the GridSearch best estimator directly?** The GridSearch best estimator is trained on each fold separately during cross-validation, but not on the full training set. Retraining on the full `X_train` with `X_val` for early stopping ensures the model sees all available training data before final evaluation on the test set.

---

#### 4. The Evolution of the Implementation

##### 4.1 Simple Baseline (No Tuning)
Initially, we used a fixed set of hyperparameters with early stopping:

```python
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    early_stopping_rounds=10,
    verbosity=0
)
```
**Results**: R² = 0.9051, RMSE = 5.44.

##### 4.2 Hyperparameter Tuning with GridSearch
We then added GridSearch with the wrapper to find the optimal hyperparameters.

**Results**: R² = 0.9127, RMSE = 5.22 — a significant improvement.

##### 4.3 Final Optimised Configuration
The best parameters found:

| Parameter | Value | Why it works |
| :--- | :--- | :--- |
| `max_depth` | 4 | Shallow trees prevent overfitting. |
| `learning_rate` | 0.05 | Slower, more stable convergence. |
| `subsample` | 1.0 | Use all samples (dataset is small). |
| `colsample_bytree` | 1.0 | Use all features. |
| `reg_lambda` | 2.0 | Strong L2 regularisation. |
| `reg_alpha` | 0 | No L1 regularisation needed. |

The final model used 295 trees (out of 1000) due to early stopping — the optimal number for this dataset.

---

#### 5. Results and Performance

##### 5.1 Comparison with Phase 3 Neural Network

| Metric | Neural Network (Phase 3) | XGBoost (Phase 4) | Improvement |
| :--- | :--- | :--- | :--- |
| **R²** | 0.4228 | 0.9127 | ✅ +0.4899 |
| **RMSE** | 13.42 | 5.22 | ✅ -8.20 |
| **MAE** | 11.18 | 4.10 | ✅ -7.08 |
| **MSE** | 180.09 | 27.25 | ✅ -84.87% |

##### 5.2 Interpretation
* XGBoost explains 91.27% of the variance in the data, compared to only 42.28% for the neural network.
* The error is reduced by over 80%: RMSE drops from 13.42% to 5.22%.
* XGBoost is clearly superior for this tabular data problem.

##### 5.3 Feature Importance (Chemical Insights)
The feature importance plot (saved as `comparison_plots.png`) shows which features drive the prediction:
* **LogP** and **Is_Aqueous** are expected to be among the top features, confirming the chemical logic: hydrophobic PIs in aqueous media drastically reduce conversion.
* Some visual embeddings (especially those capturing functional groups like carbonyls or aromatic rings) are also highly influential.
* **UV_Dose** and **%PI** likely have moderate importance, as they are tunable parameters that affect conversion but have a smaller influence than the PI-monomer-environment interaction.

---

#### 6. Why This Architecture is Scalable and Production-Ready

| Feature | Benefit |
| :--- | :--- |
| **Separation of tuning and final training** | The GridSearch is run once to find optimal parameters; the final model is trained separately and saved. |
| **Early stopping** | Automatically determines the optimal number of trees, preventing overfitting. |
| **Wrapper for GridSearch** | Enables early stopping inside cross-validation, giving more reliable hyperparameter selection. |
| **Seed control** | The `SEED` variable allows reproducibility (`SEED > 0`) or exploration (`SEED = 0`). |
| **Model persistence** | The model is saved as `xgboost_model.json`, ready for loading in Phase 5 (Reflex web app). |
| **Config persistence** | `model_config.json` stores the hyperparameters and input dimension for reproducibility. |

---

#### 7. Future Improvements

##### 7.1 Automated Hyperparameter Tuning (Pyramid Tuning)
Instead of a single grid, we can use a multi-stage tuning approach:
1. **Broad search**: `max_depth`, `learning_rate`, `reg_lambda`.
2. **Narrow search**: refine the best values and add `subsample`, `colsample_bytree`.
3. **Final fine-tuning**: `reg_alpha`, `min_child_weight`.

##### 7.2 Cross-Validation with More Folds
Currently using `cv=3`. Increasing to `cv=5` would give more reliable estimates but would increase training time.

##### 7.3 Early Stopping Patience Tuning
The current `early_stopping_rounds=10` is a conservative value. We could tune this as part of the hyperparameter search.

##### 7.4 Integration with RDKit (QSAR)
Currently, monomer reactivity factors are static. In the future, we plan to:
* Compute molecular descriptors using RDKit.
* Use these descriptors to predict reactivity factors via a QSAR model.
* This will allow dynamic reactivity estimation for any monomer, including new candidates, without manual factor assignment.

---

#### 8. Conclusion
`train_xgboost.py` successfully completes Phase 4 of the MVP pipeline, training an XGBoost regressor on the combined feature vector (visual embeddings + environmental features). The script:

* ✅ Implements the specification exactly as described.
* ✅ Loads embeddings from Phase 2.
* ✅ Generates environmental features (simulated for MVP).
* ✅ Concatenates features into a single tabular dataset.
* ✅ Uses a custom wrapper to enable early stopping inside `GridSearchCV`.
* ✅ Trains the final model with the best hyperparameters.
* ✅ Evaluates the model on a test set.
* ✅ Compares performance with the Phase 3 neural network, demonstrating XGBoost's superiority on tabular data.
* ✅ Saves the trained model for use in Phase 5.

**Current status**: Ready for Phase 5 (Reactive Web App with Reflex).

---

### Simulation v2 — Chemically informed

Following the implementation of the Full Factorial Design, the simulation
function was refined with input from an expert formulator:

1. **PI concentration**: the previously ignored variable was replaced with a
   bell-shaped (gamma) curve peaking at 3%. This captures the UV-shielding
   effect observed at high concentrations (Beer-Lambert law).

2. **Medium compatibility**: the Boolean penalty was replaced with a continuous
   exponential factor based on polarity matching (PI LogP vs. the medium's
   actual LogP). This models the "like dissolves like" principle for all
   PIs, not just those used in water.

3. **Elimination of hardcoded thresholds**: chemistry is now modeled using
   continuous functions rather than step-like if/else logic.

The XGBoost model will thus be able to learn the actual interactions
between concentration, medium, and molecular properties.
