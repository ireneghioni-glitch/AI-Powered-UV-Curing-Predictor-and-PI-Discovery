# MVP Technical Specification (v2): AI-Powered UV-Curing Predictor & PI Discovery

This document serves as the updated technical blueprint and pipeline specification for your 2-week specialization MVP [1]. It details how the original "Pneumonia Challenge" Computer Vision framework [1, 2] is fully adapted into a high-impact, chemically rigorous **"Chemical Vision"** pipeline focused on **UV-Curing Prediction & Photoinitiator (PI) Discovery** [43]. 

This version integrates detailed operational workflows for Chemical APIs, static datasets, and a hybrid sourcing strategy to ensure a theoretically complete and practically bulletproof MVP within the strict deadline [1, 5].

---

## 1. Project Overview & Scope (2-Week MVP)
The goal of this MVP is to build a complete, self-contained predictive system that evaluates the reactivity of photoinitiators and monomers under ultraviolet (UV) light, taking into account the solvent environment (aqueous vs. solvent-based) [48]. 

To guarantee a fully functional, bug-free, and mathematically sound project within the strict 2-week deadline [1, 5], we focus on the **Curing & PI Discovery** model [43]. The formulation fluid dynamics (viscosity blending models and inkjet printability Z-number) are decoupled from this sprint and reserved as a modular future expansion [48].

### Core Learning Alignment
This adaptation preserves 100% of the training syllabus requirements [1, 3]:
*   **Computer Vision (CV)**: Replacing chest X-ray pixel grids [2] with 2D chemical structure image grids. A Convolutional Neural Network (CNN) is used to extract visual feature representations from molecular diagrams [4].
*   **Supervised Learning & Regression**: Utilizing the extracted CV features alongside environmental variables to predict a quantitative target—**double bond conversion percentage (% Conversion)**—via an XGBoost regressor [19, 30].
*   **Model Evaluation**: Implementing train-test splitting, hyperparameter tuning, confusion matrices (for classification thresholds), and regression error metrics (RMSE, $R^2$) [3, 4].

---

## 2. The "Chemical Vision" Pipeline Architecture

The pipeline processes raw molecular data, extracts structural features via Computer Vision, combines them with environmental parameters, and predicts the polymerization conversion percentage.

```
+------------------------------------------------------------+
|                       INPUT DATA                           |
|  - Monomer SMILES String    - Photoinitiator SMILES String |
|  - Environment (Water vs Solvent)  - UV Energy Dose        |
+-----------------------------+------------------------------+
                              |
                              v
+------------------------------------------------------------+
|               2D MOLECULAR VISUALIZATION                   |
|  - RDKit processes SMILES into 2D chemical diagrams       |
+-----------------------------+------------------------------+
                              |
                              v
+------------------------------------------------------------+
|             COMPUTER VISION ENCODER (CNN)                  |
|  - Pre-trained CNN (Keras) processes molecular diagrams    |
|  - Extracts visual embeddings of active functional groups  |
+-----------------------------+------------------------------+
                              |
                              v
+------------------------------------------------------------+
|                 FEATURE CONCATENATION                      |
|  - Merges Monomer & PI visual embeddings with:             |
|    * Is_Aqueous (0 or 1)                                   |
|    * LogP (Hydrophobicity calculated via RDKit)            |
|    * PI Concentration (%) & UV Energy Dose (mJ/cm2)        |
+-----------------------------+------------------------------+
                              |
                              v
+------------------------------------------------------------+
|               REGRESSION MODEL (XGBoost)                   |
|  - Maps non-linear relationships of the combined vector    |
+-----------------------------+------------------------------+
                              |
                              v
+------------------------------------------------------------+
|                       PREDICTION                           |
|  - Outputs: Quantitative % Curing Conversion (0-100%)       |
+-----------------------------+------------------------------+
```

### Detailed Pipeline Steps

#### Step A: SMILES Ingestion & 2D Image Generation (Data Preprocessing)
Instead of medical DICOM/JPEG files, the raw inputs are text-based **SMILES (Simplified Molecular Input Line Entry System)** strings [7].
1.  **RDKit Ingestion**: RDKit reads the monomer (e.g., acrylic acid: `C=CC(=O)O`) and photoinitiator (e.g., Benzophenone: `O=C(c1ccccc1)c2ccccc2`) SMILES.
2.  **Visual Render**: RDKit's drawing module renders these structures into standardized, black-and-white 2D image grids of the chemical structure (e.g., $224 \times 224$ pixels, similar to standard CNN input sizes) [4].
3.  **Computer Vision Context**: This replaces OpenCV image manipulation in the original project [4]. Standardizing line thickness, atom labels, and scaling acts as the chemical preprocessing step.

#### Step B: Feature Extraction via Keras CNN Transfer Learning
To satisfy the Deep Learning core of the specialization, we use a Convolutional Neural Network (CNN) as a feature extractor [1]:
1.  **Base Network**: Import a pre-trained light-weight network (e.g., MobileNetV2 or ResNet50) from Keras Applications with frozen weights [4].
2.  **Inference**: Run the 2D molecular drawings through the CNN. The model will recognize visual patterns corresponding to chemical functional groups (e.g., detecting the $\text{C=C}$ double bonds of acrylates or the photoactive carbonyl group of benzophenones).
3.  **Visual Embeddings**: Extract the output of the global pooling layer to obtain a dense vector representation (e.g., 1024 features) for both the monomer and the photoinitiator.

#### Step C: Feature Engineering of the Mixture & Environment
The visual embeddings are concatenated with key environmental and operational features to form the final tabular input vector:
$$\text{Input Vector} = [\\text{Monomer CNN Embeddings}] + [\\text{PI CNN Embeddings}] + [\\text{Environment Variables}] + [\\text{Process Parameters}]$$

Where:
*   **`Is_Aqueous`**: Binary flag (1 for water-based, 0 for organic solvent or 100% bulk monomer) [48].
*   **`LogP`**: Octanol-water partition coefficient of the photoinitiator calculated via RDKit [43]. This serves as a proxy for solubility compatibility. If `Is_Aqueous = 1` and `LogP` is high (hydrophobic), the model learns that the photoinitiator will poorly disperse, reducing conversion efficiency.
*   **`PI_Concentration (%)`**: Weight percentage of photoinitiator in the formulation [48].
*   **`UV_Energy_Dose (mJ/cm²)`**: Total radiant energy delivered to the film [48].

#### Step D: Regression via XGBoost
The concatenated feature vector is fed into an **XGBoost Regressor** (leveraging the machine learning and regression techniques covered in your Data Analysis materials) [19, 30]. XGBoost learns the complex, highly non-linear interactions between the physical chemical structure, the solvent environment, and the energy dose to output the final predicted **Double Bond % Conversion** [19, 30, 48].

---

## 3. Data Sourcing Strategy (No Experimental Data Required)

Because you do not have access to proprietary experimental laboratory data, you can build, train, and validate this entire pipeline using highly curated, public, open-source databases [43, 48]. 

| Database / Resource | URL | Data Type | Relevance to your MVP |
| :--- | :--- | :--- | :--- |
| **MoleculeNet (DeepChem)** [46] | [deepchem.io](https://deepchem.io/) | Molecular Benchmarks | Contains standard datasets for solubility (ESOL), partition coefficients (Lipophilicity), and physical properties. Perfect for pre-training and testing chemical embeddings [46]. |
| **QM9 Dataset** [47] | [quantum-machine.org](http://quantum-machine.org/datasets/) | Quantum Chemical Properties | 134,000 organic molecules with 19 physical, thermodynamic, and electronic properties (calculated via Density Functional Theory). Ideal for training models to recognize molecular reactivity [47]. |
| **PI1M (Polymer Informatics)** [7, 44] | [github.com/RUIMINMA1996/PI1M](https://github.com/RUIMINMA1996/PI1M) | Synthetic Polymers (SMILES) | 1 million polymer structures stored in polymer-SMILES format [7, 44]. Excellent playground for performing high-throughput virtual screening and testing "invented" monomers. |
| **PolyInfo (NIMS)** [43] | [polyinfo.nims.go.jp](https://polyinfo.nims.go.jp/) | Experimental Polymer Data | World-class database containing experimental properties of polymers, including Glass Transition Temperature ($T_g$), density, and rheological parameters [43]. |
| **PubChem & ChEMBL** [47] | [pubchem.ncbi.nlm.nih.gov](https://pubchem.ncbi.nlm.nih.gov/) | Massive Chemical Registry | Ideal for bulk downloading structures of known photoinitiator classes (benzophenones, thioxanthones, acylphosphine oxides) to build your Virtual Screening library [47]. |

---

## 4. Operational Data Ingestion: APIs vs. Datasets

When building a Chemoinformatics pipeline, developers face a choice between calling live Chemical APIs and loading static Datasets. Understanding their differences and how they work together is crucial for a robust architecture.

### Operational Differences: A Comparison

| Feature | Chemical APIs (e.g., PubChem, ChEMBL) | Static Datasets (e.g., MoleculeNet, CSVs) |
| :--- | :--- | :--- |
| **Nature** | Dynamic, live web service queried programmatically. | Static, pre-compiled files (CSV, SDF, JSON) loaded locally. |
| **Data Scope** | Covers millions of registered compounds; updated constantly. | Contains a fixed, curated list of molecules and target properties. |
| **Speed** | Slow (requires network latency, HTTP requests). Subject to rate limits. | Extremely fast (disk I/O speed, fully offline-capable). |
| **Best Used For** | Real-time queries, data enrichment, and virtual screening of novel candidates. | Training models, hyperparameter tuning, and reproducible benchmarking. |

### Can one replace the other?
**No, they are distinct operational tools that serve different parts of the lifecycle.**
*   You **cannot** easily train a deep learning model solely on raw API calls because training requires loading thousands of molecules repeatedly over multiple epochs—which would trigger severe network latency and API rate-limiting blocks.
*   You **cannot** perform a virtual screening of newly modified or uncharacterized molecules using only a static dataset, because the static dataset cannot dynamically answer queries about molecules it does not contain.

### How they reconcile in a unified pipeline

In a professional data science pipeline, they are orchestrated in a **sequential, complementary loop**:

```
[Static Datasets (MoleculeNet/CSVs)] ──► Train & Validate Model (Offline)
                                                 │
                                                 ▼
[Chemical APIs (PubChem/ChEMBL)] ──────► Fetch "Anchor" Libraries (Live)
                                                 │
                                                 ▼
[RDKit Virtual Modification] ──────────► Generate Novel SMILES Candidates
                                                 │
                                                 ▼
[Trained Model (XGBoost/CNN)] ─────────► Run Inference / Predict Conversion
```

1.  **Phase 1: Model Training (Datasets)**: You use a pre-packaged, static dataset (like MoleculeNet or a downloaded CSV) containing molecular SMILES and verified experimental target labels (like Tg or solubility) to train and optimize your CNN and XGBoost models.
2.  **Phase 2: Library Anchor Fetching (APIs)**: During deployment, your pipeline uses APIs programmatically to query PubChem and download the exact SMILES and properties of established commercial photoinitiators (e.g., Benzophenone).
3.  **Phase 3: Generation & Modification (In-Silico)**: You apply virtual chemical modifications (via RDKit) to those anchor molecules to create new variations.
4.  **Phase 4: Evaluation (Model Inference)**: Your trained offline model evaluates these new, API-sourced candidates, performing high-throughput virtual screening to identify the best performers.

---

## 5. Python API Integration & Core Libraries

To fetch molecular data programmatically, your pipeline will use Python packages that wrap REST APIs. This allows you to gather compounds, look up property values, and search by chemical structure.

### A. PubChemPy (PubChem API Wrapper)
PubChemPy allows you to interact with PubChem’s PUG REST API. You can search by chemical name, SMILES, or perform substructure searches.

```python
import pubchempy as pcp

# 1. Fetch a specific photoinitiator by name
compound = pcp.get_compounds('Benzophenone', 'name')[0]
print(f"SMILES: {compound.isomeric_smiles}")
print(f"XLogP (Hydrophobicity): {compound.xlogp}")
print(f"Molecular Weight: {compound.molecular_weight}")

# 2. Perform a substructure search to find related photoinitiators
# This searches for all molecules containing the benzophenone active core
benzophenone_smiles = 'C1=CC=C(C=C1)C(=O)C2=CC=C(C=C1)'
related_compounds = pcp.get_compounds(benzophenone_smiles, searchtype='substructure')
print(f"Found {len(related_compounds)} compounds containing the active carbonyl-phenyl core.")
```

### B. ChEMBL Webresource Client
ChEMBL offers a structured client to query biological and chemical targets. It is highly structured and useful for retrieving targeted compound families.

```python
from chembl_webresource_client.new_client import new_client

# Search ChEMBL for photoactive molecules or search by keyword
molecule_api = new_client.molecule
results = molecule_api.filter(molecule_synonyms__molecule_synonym__icontains='benzophenone')
print(f"ChEMBL returned {len(results)} matches for benzophenone synonyms.")
```

---

## 6. Recommended MVP Sourcing Strategy (The 2-Week Hack)

To ensure you have a "theoretically and practically finished" project in less than two weeks without running into API rate-limiting errors or missing data, we implement a **Hybrid Sourcing Strategy**:

1.  **Step 1: Download a Static Core CSV (Day 1)**: 
    Export a curated dataset of monomer properties and polymer $T_g$ values directly from PolyInfo or open-source GitHub repositories (such as the PI1M repository or MoleculeNet's ESOL dataset). This acts as your secure training foundation.
2.  **Step 2: Programmatic API Enrichment (Days 2-3)**: 
    Write a lightweight Python script using `PubChemPy` that reads your CSV file, takes the SMILES column, queries PubChem's API to fetch any missing physical descriptors (such as XLogP, density, or polar surface area), and writes them back into a consolidated CSV. 
3.  **Step 3: Train & Validate Offline (Days 4-9)**:
    Perform all training, validation, and hyperparameter tuning offline using the enriched, static CSV. This ensures immediate feedback loops and zero dependencies on internet connectivity during model tuning.
4.  **Step 4: Live API Integration in Streamlit App (Days 10-12)**:
    In your final Streamlit GUI, allow users to type a chemical name (e.g., "TPO"). Under the hood, your app will call `pubchempy.get_compounds(name, 'name')` to retrieve its SMILES on-the-fly, convert it to a 2D drawing via RDKit, and pass it directly to your trained model for reactivity prediction.

---

## 7. Two-Week MVP Implementation Timeline (Syllabus Mapping)

To ensure success, your sprint is modeled directly after the "Pneumonia Challenge" steps [4]:

*   **Days 1-3: Data Collection & Curation** [4]
    *   Download known photoinitiator and monomer SMILES from PubChem [47].
    *   Set up RDKit in your python environment to generate 2D molecular drawings.
*   **Days 4-6: Model Setup & Transfer Learning** [4]
    *   Import a pre-trained CNN in Keras [4].
    *   Pass the RDKit drawings through the frozen base network to extract embeddings [4].
    *   Set up your tabular dataset by merging the CNN features with the environment (`Is_Aqueous`), `LogP`, and `UV Dose` features.
*   **Days 7-9: Regressor Training & Hyperparameter Tuning** [4]
    *   Train your XGBoost model on the compiled dataset.
    *   Perform a hyperparameter grid search (learning rate, max depth, estimators) to optimize the performance [4].
*   **Days 10-12: Evaluation & GUI Development** [3, 4]
    *   Plot the performance metrics (Predicted vs. Actual plots, ROC curves for a defined threshold, Confusion Matrix for >85% conversion success) [3, 4, 9].
    *   Build a simple local **Streamlit** app where users can input an SMILES string, view the 2D molecule (CV output), and get an instant predicted conversion rate.
*   **Days 13-14: GitHub Delivery** [3]
    *   Push your structured OOP code to GitHub [3].
    *   Write a professional README including installation, chemical background, and the future modular roadmap (incorporating the formulation model and Optuna optimization) [3].
