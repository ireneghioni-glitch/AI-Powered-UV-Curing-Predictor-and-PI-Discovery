# Phase 4 (v2) — Development Log
## From 8,960 Random Rows to a 520,000-Row Design of Experiments

### Document Purpose
This document chronicles the evolution of the Phase 4 model from its initial "one random row per PI–monomer pair" dataset to a structured full-factorial design with reduced dimensionality and a chemistry-aware target simulation. It also covers the addition of CAS number support and the UI adjustments that followed.

The document is written for future reference and for anyone joining the project: it explains why each change was necessary, what was changed, and what theory underlies each decision.

---

## 1. The Starting Point (v1)

The first version of Phase 4 produced a dataset of 8,960 rows: one row for each (PI, monomer) pair, with four environmental features sampled at random.

### Problems with v1
* **Problem 1 — Randomness in the target simulation.** The function `simulate_conversion` called `np.random.uniform(75, 95)` for every row. *Consequence:* two rows describing the same PI–monomer pair received different base conversion values. The model could not learn how environmental conditions affected the outcome, because the underlying chemistry signal was buried in noise.
* **Problem 2 — One row per pair, random conditions.** The dataset did not systematically explore the environmental space. Each pair was tested under one random combination of (dose, LogP, concentration, environment). *Consequence:* the model never saw the same pair under different conditions, so it could not learn the interaction between molecule identity and process parameters.
* **Problem 3 — Underutilized features.** Some environmental variables (like `pi_concentration`) were present in the feature matrix but never used in the target simulation. The model correctly learned that they were irrelevant — which was chemically false.
* **Problem 4 — High-dimensional input.** Each row contained 2,564 features: 1,280 from the PI embedding, 1,280 from the monomer embedding, and 4 environmental parameters. With only 8,960 rows, the ratio of features to samples was already stretched. Adding more rows (see Section 2) would have required 92 GB of RAM if kept as float64.

---

## 2. Full Factorial Design (Design of Experiments)

### 2.1 Motivation
A chemist designing a formulation campaign does not test one random condition per molecule. They design an experiment matrix: the same molecule is tested under a systematic grid of concentrations, doses, and environments. This is called a **Design of Experiments (DoE)**, and the simplest version is a full factorial design: enumerate every combination of the factors of interest.

### 2.2 What We Did
For every (PI, monomer) pair, we generated 1,000 environmental conditions:

| Factor | Levels |
| :--- | :--- |
| **Environment (`is_aqueous`)** | 2 (Solvent / Water) |
| **LogP** | 5 values from 0.0 to 6.0 |
| **PI concentration** | 10 values from 0.5% to 5.0% |
| **UV dose** | 10 values from 100 to 1000 mJ/cm² |

$$	ext{Total: } 2 	imes 5 	imes 10 	imes 10 = 1,000 	ext{ conditions per pair.}$$

With 560 unique (PI, monomer) pairs (after augmentation averaging — see Section 4), the dataset became:
$$	ext{560 pairs} 	imes 1,000 	ext{ conditions} = 560,000 	ext{ rows}$$

### 2.3 Why It Matters
The model now sees each pair 1,000 times, each time under a different condition. It can learn:
* *"For this pair, increasing dose helps up to ~400 mJ/cm², then saturates"*
* *"For this pair, moving from solvent to water reduces conversion by X%"*
* *"For this pair, the optimum PI concentration is around 3%"*

None of this was learnable in v1.

---

## 3. Deterministic Pair-Specific Simulation

### 3.1 The Problem
With the factorial expansion, the same (PI, monomer) pair appears in 1,000 rows. If the "base reactivity" of that pair changed randomly between rows, the model would see:
```text
Row 1: TPO-L + TMPTA + dose=100 -> base=82.3
Row 2: TPO-L + TMPTA + dose=200 -> base=77.9   ← different!
Row 3: TPO-L + TMPTA + dose=300 -> base=91.4   ← different!
```
The model would attribute variation to environmental conditions when it actually came from the base. It would never learn the true effect of dose.

### 3.2 The Solution: Hash-Based Deterministic Seeding
We use MD5 hashing to convert a (PI, monomer) name pair into a stable 32-bit integer, which we use as the seed of a NumPy random generator:

```python
import hashlib

def _stable_pair_seed(pi_name, monomer_name):
    key = f"{pi_name}|{monomer_name}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest()[:8], 16)
```

This produces the same integer every time, on every machine, forever. The RNG created from it produces the same base value. The pair (TPO-L, TMPTA) always has the same base reactivity, regardless of which row we are computing.

### 3.3 Why "Hash" and Not Python's Built-In `hash()`?
Python randomizes `hash()` of strings per process for security reasons. Two runs of the same script would produce different seeds. `hashlib.md5()` is deterministic across processes and machines, which is exactly what we need.

### 3.4 Theoretical Concept: "Deterministic Randomness"
This technique is called **deterministic randomness** or **pseudo-random with reproducible seed**. The output looks random (different pairs have different values), but it is fully reproducible given the same input. It is the standard technique for any simulation that must be reproducible.

---

## 4. Embedding Reduction: Averaging + PCA

### 4.1 Problem: RAM Explosion
The naïve factorial dataset would have shape:
$$	ext{560,000 rows} 	imes 2,564 	ext{ features} 	imes 8 	ext{ bytes (float64)} = 11.5 	ext{ GB}$$
Plus copies during train/val/test split, plus copies for GridSearchCV folds. Peak RAM: ~30-40 GB. Not feasible on a standard laptop.

### 4.2 Fix A — Averaging Over Augmentations
The PI embeddings had 208 rows because they contained 52 molecules $	imes$ 4 rotations (original + $90^\circ$ + $180^\circ$ + $270^\circ$). The monomer embeddings had 40 rows for 10 molecules $	imes$ 4 rotations.

The 4 rotations of the same molecule are not 4 different molecules. They are the same molecule viewed at different angles — a data augmentation used during CNN training. For XGBoost, feeding the same molecule 4 times with slight variations creates a pseudo-replicated dataset that biases the model.

**Fix:** for each molecule, average its 4 rotation-embeddings into a single embedding.

```python
import numpy as np

def average_augmentations(embeds, meta):
    unique_names = meta["name"].unique()
    return np.array([
        embeds[meta["name"] == name].mean(axis=0)
        for name in unique_names
    ])
```

**Result:** PI embeddings go from $(208, 1280)$ to $(52, 1280)$. Monomer embeddings go from $(40, 1280)$ to $(10, 1280)$.

Dataset size after averaging: $52 	ext{ PIs} 	imes 10 	ext{ monomers} 	imes 1,000 	ext{ conditions} = 520,000 	ext{ rows}$.

### 4.3 Fix B — Principal Component Analysis (PCA)

#### 4.3.1 The Problem: Too Many Features, Too Much Redundancy
Each molecule was still represented by a 1,280-dimensional embedding. But 1,280 numbers to describe a single molecule is overkill. Most of those dimensions are:
* **Redundant:** different dimensions carry nearly the same information.
* **Noisy:** small variations due to image rendering artifacts, not chemistry.
* **Correlated:** the CNN learned to extract many overlapping features.

Having 1,280 features when only ~50 carry meaningful chemical information creates the **curse of dimensionality**: the model needs exponentially more data to learn reliably as the number of features grows.

#### 4.3.2 What PCA Is — An Intuitive Explanation
Imagine you are looking at a photograph of a cloud of points floating in 3D space. The cloud is shaped like a flat, elongated pancake: it's spread out in two directions but very thin in the third. To describe each point, you currently use 3 coordinates $(x, y, z)$. But because the cloud is nearly flat, you could describe each point with just 2 coordinates if you rotated the axes to align with the pancake.

#### 4.3.3 The Math in One Paragraph
PCA computes the covariance matrix of the data (which measures how each pair of features varies together), then finds its eigenvectors and eigenvalues. Each eigenvector is a principal component; each eigenvalue tells you how much variance that component explains. You then keep the top $N$ components that together explain, say, 95% of the total variance.

#### 4.3.4 What We Gain and What We Lose

| Aspect | Before PCA | After PCA |
| :--- | :--- | :--- |
| **Features per PI** | 1,280 | 51 |
| **Features per monomer** | 1,280 | 9 |
| **Total features in X** | 2,564 | 64 |
| **RAM (520K rows)** | ~10.7 GB | ~135 MB |
| **Information lost** | 0% | ~0% (in this case) |
| **Redundancy removed** | None | Massive |

*Important note on our specific case:* because we had only 52 PI samples, PCA with 51 components (the maximum allowed) captured 100% of the variance. We did not actually lose information. We just removed the redundant dimensionality.

#### 4.3.5 Why 51 and 9 Components?
PCA cannot produce more components than $\min(n_{	ext{samples}}, n_{	ext{features}}) - 1$. With 52 PIs, the cap is 51. With 10 monomers, the cap is 9. Using the maximum keeps all information while removing the redundancy.

#### 4.3.6 The Training/Inference Contract
A PCA model is learned from training data. To apply the same transformation at inference time, we must save the fitted PCA models to disk and reload them:

```python
import joblib

# Save during training
joblib.dump(pca_pi, BASE_DIR / "pca_pi.pkl")

# Reload at inference
pca_pi = joblib.load("phase4/pca_pi.pkl")
pi_emb = pca_pi.transform(pi_emb_raw.reshape(1, -1))[0]
```

This is a critical ML principle: **any transformation learned from training data (scalers, encoders, PCA, normalizers) must be saved and reapplied identically at inference time.** Otherwise, the inputs the model sees in production differ from the ones it was trained on, and predictions degrade silently.

---

## 5. Chemistry-Aware Target Simulation

### 5.1 Why Refine the Simulation?
The original simulation used three factors:
$$	ext{conversion} = 	ext{base} 	imes 	ext{monomer\_factor} 	imes 	ext{dose\_factor} 	imes 	ext{penalty}$$
with `penalty` being a hard step function (1.0 or 0.3 or 0.5 depending on LogP thresholds).

Three issues with the original version:
1. `pi_concentration` was ignored — it existed as a feature but had no effect on the target. The model would learn it was irrelevant, which is chemically false.
2. The penalty was discontinuous — hard thresholds at $	ext{LogP} = 3.0$ and $4.0$ created abrupt jumps that don't exist in real chemistry.
3. Only aqueous systems were penalized — a hydrophilic PI in an organic solvent was not penalized, though solubility mismatch is a general phenomenon, not aqueous-specific.

### 5.2 The 5-Factor Multiplicative Model
The simulation was rewritten as:
$$	ext{conversion} = 	ext{base} 	imes 	ext{monomer\_factor} 	imes 	ext{dose\_factor} 	imes 	ext{pi\_factor} 	imes 	ext{medium\_factor}$$

Each factor captures a distinct physical effect:

* **Factor 1 — UV dose saturation (`dose_factor`):**
  More photons $ightarrow$ more radicals $ightarrow$ more conversion. But saturating:
  $$	ext{dose\_factor} = 1 - e^{-0.01 	imes 	ext{dose}}$$
  At $	ext{dose} = 0$, $	ext{factor} = 0$. At $	ext{dose} ightarrow \infty$, $	ext{factor} ightarrow 1$.

* **Factor 2 — PI concentration (`pi_factor`):**
  A bell curve with an optimum around 3%:
  $$	ext{pi\_factor} = \left(rac{c}{c_{	ext{opt}}}ight) 	imes e^{1 - c / c_{	ext{opt}}} \quad 	ext{where } c_{	ext{opt}} = 3.0$$
  Below the optimum, radicals are scarce. Above, Beer-Lambert screening dominates: the UV light is absorbed by the PI at the surface and never reaches the bulk. This behavior is well documented in photopolymerization kinetics.

* **Factor 3 — Medium compatibility (`medium_factor`):**
  "Like dissolves like": the PI's LogP should match the medium's effective LogP.
  
  | Medium | Effective LogP |
  | :--- | :--- |
  | **Organic solvent (typical acrylate systems)** | 4.0 |
  | **Water** | -1.0 |
  
  $$	ext{medium\_factor} = e^{-0.15 	imes |	ext{LogP}_{	ext{PI}} - 	ext{LogP}_{	ext{medium}}|}$$
  If the PI is a perfect match for the medium, $	ext{factor} = 1.0$. If they are very mismatched (e.g., a highly hydrophobic PI in water), factor drops to ~0.4. The transition is continuous — no hard thresholds.

### 5.3 Why This Matters for the Model
With a continuous, chemistry-aware simulation, the XGBoost model learns:
* The shape of the concentration response (bell curve, not linear)
* The shape of the dose response (saturation, not linear)
* The continuous effect of medium mismatch

This is far more useful than learning three hard thresholds. If real experimental data becomes available, the same model architecture will work — only the target values will change, not the model structure.

---

## 6. Training/Inference Alignment — The Bug We Hit and Fixed

### 6.1 The Bug
After training v2 with PCA, the model expected 64 features. But `inference/pipeline.py` still computed 2,564 features (raw embeddings, no averaging, no PCA). When the user clicked "Predict" in the Reflex app, the model received 2,564 numbers and returned:
```text
Prediction failed: Feature shape mismatch, expected: 64, got 2564
```

### 6.2 The Fix
Three changes in `inference/pipeline.py`:
* **Fix 1 — Save PCA models in the training script:**
  ```python
  joblib.dump(pca_pi, BASE_DIR / "pca_pi.pkl")
  joblib.dump(pca_mono, BASE_DIR / "pca_mono.pkl")
  ```
* **Fix 2 — Load PCA models at inference time as lazy singletons** (same pattern as TensorFlow and XGBoost).
* **Fix 3 — Apply the same 4-rotation averaging at inference that was applied at training:**
  ```python
  def smiles_to_averaged_embedding(smiles: str) -> np.ndarray:
      # Compute 4 embeddings (orig + 90° + 180° + 270°), average them ...
  ```

### 6.3 Trade-off
Inference is now $4	imes$ slower per molecule (4 embeddings instead of 1), but the input distribution matches training exactly. A small latency cost in exchange for correctness is the right trade-off.

### 6.4 The General Lesson
Any preprocessing step learned from training data must be (a) saved, and (b) applied identically at inference.
This includes: scalers, encoders, PCA, normalization parameters, tokenizers, and any augmentation strategy that affects the effective input distribution.

---

## 7. CAS Number Support

### 7.1 The Feature Request
Chemists use CAS Registry Numbers as the standard identifier for molecules. For example, Benzophenone is CAS 119-61-9; Acrylic acid is 79-10-7. The app should accept CAS numbers interchangeably with names.

### 7.2 Why It Worked "For Free"
The PubChem API endpoint `/compound/name/{query}/cids/` does not only accept names — it accepts any registered synonym, including CAS numbers, trade names, IUPAC names, and registry identifiers.

Because the existing `PubChemClient` already uses this endpoint, no code changes to the backend were needed. A CAS number like 119-61-9 resolves exactly as Benzophenone would, and returns the same SMILES.

### 7.3 The Limitation We Discovered
Not every CAS is registered in PubChem. Compound TMPTMA (Trimethylolpropane trimethacrylate, CAS 3290-92-4) is a common specialty monomer that is not in PubChem's synonym database. Neither its CAS nor its common abbreviation resolved via the API.

**Solution:** extend the manual SMILES dictionary with the molecule and its aliases:
```python
MANUAL_SMILES = {
    # ...
    "TMPTMA": "<smiles>",
    "3290-92-4": "<smiles>",
    "Trimethylolpropane trimethacrylate": "<smiles>",
}
```
Three keys for the same molecule — the client matches case-insensitively, so any input form works.

### 7.4 The General Pattern

| Input type | Likelihood of PubChem resolution |
| :--- | :--- |
| **IUPAC name** | Very high |
| **Common CAS (Benzophenone, Acrylic acid)** | Very high |
| **Specialty monomer CAS** | ~30–50% |
| **Trade abbreviation (TMPTA, HDDA)** | Sometimes |
| **Niche abbreviation (TMPTMA)** | Rarely |

**Takeaway:** for any chemistry-facing tool, PubChem alone is not sufficient. A curated manual dictionary for the domain's most-used molecules is a required safety net. This is standard practice in industrial cheminformatics.

### 7.5 UI Change
Updated the input placeholders and labels in `app/pages/index.py`:
```python
rx.text("Photoinitiator name or CAS number", font_weight="bold"),
rx.input(
    name="pi_name",
    placeholder="e.g. Benzophenone or 119-61-9",
    ...
),
```
Also updated the error banner to suggest trying the chemical name if a CAS lookup fails.

---

## 8. Results

### 8.1 Model Performance

| Metric | v1 | v2 |
| :--- | :--- | :--- |
| **Training rows** | 8,960 | 520,000 |
| **Features** | 2,564 | 64 |
| **$R^2$** | 0.9127 | 0.9860 |
| **RMSE** | 5.22 | 1.58 |
| **MAE** | 4.10 | 1.15 |
| **Best trees** | 295 | 999 (hit limit) |

The MAE of 1.15 percentage points is below the accuracy of typical FTIR measurement (~2–3 points), suggesting the model fits the simulation extremely well.

*Caveat:* the target is still simulated. High $R^2$ and low MAE mean the model has learned our chemistry simulation, not real-world curing behavior. The metrics represent upper-bound performance on synthetic data, not predictive accuracy on experimental systems.

### 8.2 What Changed in Production
The Reflex app now:
* Accepts both names and CAS numbers as input
* Uses the v2 model (64 features via PCA)
* Applies 4-rotation averaging at inference to match training
* Correctly separates the two lookup paths (PI vs monomer)

---

## 9. Lessons Learned

* **A. Randomness in the target is a bug, not a feature.**
  If the simulated target contains randomness unrelated to the input features, the model will learn nothing useful. Every simulation of this kind must produce deterministic targets given deterministic inputs. Use hash-based seeding when the simulation must look realistic but be reproducible.

* **B. Design of Experiments beats random sampling.**
  A factorial design gives the model systematic coverage of the input space. Random sampling gives accidental coverage. When the number of conditions is manageable (hundreds to thousands), factorial is almost always the right choice.

* **C. Dimensionality reduction is not optional at scale.**
  At 2,564 features and 520,000 rows, the dataset would not fit in RAM on a normal machine. PCA is not a "nice to have" — it is what makes the pipeline physically runnable. Any ML pipeline that scales must include a dimensionality-reduction step.

* **D. PCA is a rotation, not a selection.**
  PCA does not "throw away" specific features. It rotates the coordinate system to align with the directions of maximum variance, then keeps the rotated axes that matter most. The information is redistributed, not discarded.

* **E. Preprocessing is a contract.**
  Any transformation learned from training data — PCA, scaling, encoding — must be saved and reapplied identically at inference. Silent mismatches cause silent degradation. Making the contract explicit (by serializing the transformer) is the professional solution.

* **F. PubChem is not enough.**
  PubChem is a wonderful resource but incomplete for specialized domains. Any chemistry-facing tool needs a curated domain-specific dictionary as a safety net. This is not a flaw of the tool — it is a design requirement of the domain.

* **G. Small UI changes have outsized impact.**
  Changing two input placeholders to advertise CAS support took five minutes and quadrupled the usefulness of the app for a real chemist. Effort is not proportional to impact.

---

## 10. Conclusion

The v2 model is a substantially different — and substantially better — system than v1. It trades RAM for structure, randomness for determinism, and hard thresholds for continuous chemistry.

The four changes (factorial expansion, deterministic simulation, PCA reduction, CAS support) are independent technically but complementary strategically: each was necessary to make the others viable, and together they turn a working MVP into a defensible demonstration of applied machine learning in an industrial chemistry context.

### Future Improvements Already Scoped:
* Increase `n_estimators` from 1,000 to 3,000 (`best_iteration` hit the limit)
* Re-run hyperparameter tuning (GridSearchCV or Optuna) on the new dataset
* Replace static monomer factors with QSAR-derived reactivity
* Fetch and store CAS numbers as a separate column
* Validate against real FTIR measurements from a collaborating lab
