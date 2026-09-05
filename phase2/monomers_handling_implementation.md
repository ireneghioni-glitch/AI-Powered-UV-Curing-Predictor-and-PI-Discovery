# Extending the Pipeline: Integrating Monomers into Phase 1 and Phase 2

## The Missing Piece: Why Monomers Matter

In the original MVP specification, the UV‑Curing predictor was designed to evaluate the reactivity of **photoinitiators (PIs)** under UV light, taking into account environmental parameters such as solvent type and UV dose.

However, during the development of Phase 3 (The Deep Learning Core), a critical question emerged:

> *"To predict curing conversion, shouldn't we also know which monomer (or monomer blend) is being used?"*

**The answer is yes.** A photoinitiator alone does not polymerise; it needs monomers (or oligomers) to react with. The structure of the monomer – its functionality, polarity, steric hindrance, and type of double bond – **dramatically affects** the polymerisation kinetics and the final conversion percentage.

| **Monomer characteristic** | **Effect on conversion** |
|----------------------------|--------------------------|
| **Functionality** (number of double bonds) | Multifunctional monomers (e.g., TMPTA, tri‑acrylate) crosslink faster and reach higher conversions than monofunctional ones (e.g., butyl acrylate). |
| **Size and branching** | Bulky monomers slow down radical diffusion, reducing conversion. |
| **Polarity** (OH, COOH groups) | Affects PI solubility and compatibility with the solvent. |
| **Double‑bond type** | Acrylates polymerise faster than methacrylates (due to steric hindrance of the methyl group). |

**Without the monomer**, the model would be trying to predict conversion knowing only the PI and the environment – like trying to predict a car's speed knowing only the fuel type, without knowing whether it's a city car or a Formula 1 racing machine.

---

## The Decision: Expand the Dataset

We decided to **extend the pipeline** to include monomers alongside PIs. This meant:

1. **Repeating Phase 1** (SMILES fetch + image generation) for a curated list of monomers.
2. **Repeating Phase 2** (embedding extraction) for the monomer images.
3. **Combining** PI embeddings and monomer embeddings (plus environmental features) to form the final feature vector for the regressor.

---

## Phase 1 Extension: Fetching Monomer SMILES and Generating Images

### Step 1: Selecting a Representative Monomer List

We chose **10 monomers** covering a wide range of functionalities and properties:

| Trade Name | IUPAC Name | Functionality | Type |
|------------|------------|---------------|------|
| TMPTA | Trimethylolpropane triacrylate | Tri‑acrylate | Highly reactive, cross‑linking |
| DEGDA | Diethylene glycol diacrylate | Di‑acrylate | Reactive, flexible |
| HDDA | 1,6‑Hexanediol diacrylate | Di‑acrylate | Reactive, hydrophobic |
| PEGDA | Poly(ethylene glycol) diacrylate | Di‑acrylate | Hydrophilic, flexible |
| HEMA | 2‑Hydroxyethyl methacrylate | Mono‑methacrylate | Hydrophilic, polar |
| MMA | Methyl methacrylate | Mono‑methacrylate | Classic, glassy |
| Butyl acrylate | Butyl acrylate | Mono‑acrylate | Flexible, hydrophobic |
| Acrylic acid | Acrylic acid | Mono‑acrylate | Polar, acidic |
| Styrene | Styrene | Mono‑styrene | Aromatic, slow |
| IBOA | Isobornyl acrylate | Mono‑acrylate | Bulky, high Tg |

**Rationale**:
- **TMPTA** represents highly cross‑linking monomers (high conversion).
- **DEGDA / HDDA** represent di‑functional systems.
- **PEGDA** tests the effect of a hydrophilic environment.
- **HEMA** represents polar, mono‑functional systems.
- **MMA / Butyl acrylate** compare methacrylate (slow) vs acrylate (fast).
- **Acrylic acid** tests extreme polarity.
- **Styrene** introduces an aromatic system.
- **IBOA** tests the steric effect.

### Step 2: Fetching SMILES (fetch_molecules_monomers.py)

We adapted the existing `fetch_molecules_PIs.py` script to work with monomers:

- **Input**: `monomers_list.csv` (names only).
- **Process**: The script tries to find the SMILES via:
  1. **Cache** (`smiles_cache_monomers.csv`) – avoids re‑fetching.
  2. **Manual fallback** (`manual_smiles_monomers` dictionary) – for the 10 common monomers.
  3. **PubChem REST API** – for new monomers not in the fallback list.
- **Output**: `molecules_monomers.csv` (name, SMILES, role).

**Why cache?** To avoid duplicate processing and repeated API calls when the list grows.

**Why manual fallback?** To guarantee 100% success for the most common monomers without relying on network calls.

### Step 3: Generating Images (generate_images_monomers.py)

We adapted the existing `generate_images_PIs.py` script:

- **Input**: `molecules_monomers.csv`.
- **Process**: For each SMILES:
  1. Convert to 2D molecular drawing using **RDKit**.
  2. Convert to **grayscale** (1 channel, 224×224).
  3. Apply **data augmentation** (rotations 90°, 180°, 270°).
- **Output**:
  - `molecular_images_monomers.npz` (40 images: 10 monomers × 4 augmentations).
  - `molecular_metadata_monomers.csv` (metadata: name, SMILES, role, augment).

---

## Phase 2 Extension: Extracting Monomer Embeddings

We adapted the existing `extract_embeddings_PIs.py` script to work with monomer images:

- **Input**: `molecular_images_monomers.npz` and `molecular_metadata_monomers.csv`.
- **Process**:
  1. Load images (40 images, 224×224, grayscale).
  2. Preprocess for MobileNetV2 (grayscale → RGB, normalise to [-1, 1]).
  3. Load **MobileNetV2** with ImageNet weights (frozen).
  4. Apply **Global Average Pooling** to obtain a 1280‑dimensional vector per image.
- **Output**:
  - `embeddings_monomers.npy` (40 × 1280).
  - `embeddings_metadata_monomers.csv` (metadata).

**Why the same architecture?** Using MobileNetV2 for both PIs and monomers ensures that embeddings are generated in the **same feature space**, making them compatible for concatenation in the final dataset.

---

## The Evolution of the Pipeline

### Before (Only PIs)

```text
[PI Names] → fetch_molecules_PIs.py → [PI SMILES CSV]
           → generate_images_PIs.py → [PI Images NPZ]
           → extract_embeddings_PIs.py → [PI Embeddings]

[PI Embeddings] + [Environmental features] → Regressor → Prediction
```

### After (PIs + Monomers)

```text
[PI Names]   → fetch_molecules_PIs.py → [PI SMILES CSV]
             → generate_images_PIs.py → [PI Images NPZ]
             → extract_embeddings_PIs.py → [PI Embeddings]

[Monomer Names] → fetch_molecules_monomers.py → [Monomer SMILES CSV]
                → generate_images_monomers.py → [Monomer Images NPZ]
                → extract_embeddings_monomers.py → [Monomer Embeddings]

[PI Embeddings] + [Monomer Embeddings] + [Environmental features] → Regressor → Prediction
```

**The final feature vector becomes**:

```text
[1280 PI features] + [1280 Monomer features] + [Is_Aqueous, LogP, %PI, UV_Dose] = 2564 features
```

---

## Why This Matters

| **Before (PI only)** | **After (PI + Monomer)** |
|----------------------|--------------------------|
| Conversion depends only on the PI and environment. | Conversion depends on PI, monomer, and environment. |
| The model ignores the chemistry of the monomer. | The model learns the interaction between PI and monomer. |
| Predictions are less realistic and less generalisable. | Predictions are more chemically informed and robust. |

**With this extension, the model can now learn:**

- Which PI works best with which monomer.
- How monomer functionality affects conversion.
- How the combination of PI and monomer responds to different environments (solvent vs. aqueous).

---

## Summary of New Files

| File | Description |
|------|-------------|
| `phase1/fetch_molecules_monomers.py` | Fetches SMILES for monomers. |
| `phase1/generate_images_monomers.py` | Generates 2D images for monomers. |
| `phase1/data/molecules_monomers.csv` | SMILES of 10 monomers. |
| `phase1/data/molecular_metadata_monomers.csv` | Metadata for monomer images (40 rows). |
| `phase1/images/molecular_images_monomers.npz` | 40 monomer images (10 × 4 augmentations). |
| `phase2/extract_embeddings_monomers.py` | Extracts embeddings for monomer images. |
| `phase2/data/embeddings_monomers.npy` | 40 × 1280 embedding vectors. |
| `phase2/data/embeddings_metadata_monomers.csv` | Metadata for monomer embeddings. |

---

## How to Add New Monomers in the Future

To expand the monomer dataset:

1. **Add the monomer** to `molecules_config_monomers` in `fetch_molecules_monomers.py`:
   ```python
   {"primary_names": ["NewMonomer"], "alt_name": "IUPAC name", "role": "monomer"}
   ```

2. **Optionally**, add its SMILES to the `manual_smiles_monomers` dictionary (to avoid API calls).

3. **Re‑run** the three scripts:
   ```batch
   python phase1/fetch_molecules_monomers.py
   python phase1/generate_images_monomers.py
   python phase2/extract_embeddings_monomers.py
   ```

The cache (`smiles_cache_monomers.csv`) will prevent duplicate processing of existing monomers.

---

## Current Status

| **Component** | **Status** |
|---------------|------------|
| PI SMILES (56 molecules) | ✅ Done |
| Monomer SMILES (10 molecules) | ✅ Done |
| PI images (224 images) | ✅ Done |
| Monomer images (40 images) | ✅ Done |
| PI embeddings (224 × 1280) | ✅ Done |
| Monomer embeddings (40 × 1280) | ✅ Done |
| Combined dataset (PI + Monomer + Features) | ⏳ Next step |

---

## Next Step: Combining Embeddings and Training the Regressor

Now that we have embeddings for both PIs and monomers, the next phase will:

1. **Create a combined dataset** by pairing each PI embedding with each monomer embedding (i.e., all possible PI–monomer combinations).
2. **Add environmental features** (Is_Aqueous, LogP, PI_Concentration, UV_Dose).
3. **Train the regressor** (neural network or XGBoost) on the combined feature vectors.

This will allow the model to predict conversion for **any PI–monomer pair** under any condition.

---

**This extension transforms the MVP from a PI‑centric predictor into a true formulation‑aware predictive tool.**
