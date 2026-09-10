# Project Proposal and Personalized Study Plan (v4): AI for UV-Curable and Inkjet Formulations Chemistry

This document represents the comprehensive version (v4) of the project proposal and study roadmap for your specialization. It fully integrates the **Photoinitiator Discovery and Environmental Impact** model with the **Formulation-Side Prediction (Viscosity, Surface Tension, and Jetting Performance)**, concluding with the interactive deployment architecture.

---

## 1. The Dual-Model Capstone Project: Curing vs. Printing Performance

A successful UV inkjet ink must satisfy two independent, highly complex physical criteria simultaneously [48]:
1. **Printing Performance (Jetting Window)**: The liquid ink must be stable and easily jetted through micro-nozzles without clogging or forming satellite droplets. This is dictated by physical properties: **viscosity** (ideally 8-15 cPs at the jetting temperature) and **surface tension** (ideally 25-35 mN/m) [48].
2. **Curing Performance (Polymerization Conversion)**: Once jetted onto the substrate, the ink must cure instantly under UV light, reaching a high double-bond conversion percentage ($\% \text{ conversion}$) to ensure proper adhesion, chemical resistance, and safety [48].

To accomplish your goals, the project will implement a **Dual-Model Pipeline**:

```
                               ┌──────────────────────────────────────────┐
                               │       FORMULATION RATIOS & COMPONENTS    │
                               │ (Monomers, Photoinitiators, Solvents/H2O)│
                               └────────────────────┬─────────────────────┘
                                                    │
                   ┌────────────────────────────────┴────────────────────────────────┐
                   ▼                                                                 ▼
┌──────────────────────────────────────┐                          ┌──────────────────────────────────────┐
│        MODEL 1: PRINTING SENSORS     │                          │         MODEL 2: CURING PREDICTOR    │
│    (Viscosity & Surface Tension)     │                          │     (% Double Bond Conversion)       │
└──────────────────┬───────────────────┘                          └──────────────────┬───────────────────┘
                   │                                                                 │
                   ▼                                                                 ▼
┌──────────────────────────────────────┐                          ┌──────────────────────────────────────┐
│  • Mixture Viscosity (Target: <15 cP)│                          │  • % Conversion (Target: >85%)       │
│  • Surface Tension (Target: 30 mN/m) │                          │  • Tg of the cured film [43]         │
│  • Jetting Window (Z-Number / Oh)    │                          │  • Curing speed insights             │
└──────────────────┬───────────────────┘                          └──────────────────┬───────────────────┘
                   │                                                                 │
                   └────────────────────────────────┬────────────────────────────────┘
                                                    │
                                                    ▼
                               ┌──────────────────────────────────────────┐
                               │      OPTUNA BAYESIAN OPTIMIZATION        │
                               │   (Finds the perfect ink composition)    │
                               └──────────────────────────────────────────┘
```

---

## 2. Model 1: Formulation-Side Prediction (Properties & Jetting Performance)

Because commercial formulation data is highly proprietary, we will build this model using a hybrid physical/machine-learning approach [48]:

### A. Feature Engineering of the Mixture
The input to Model 1 is a tabular row representing the **mass fractions ($w_i$)** of each component in your ink and their individual physical properties [48]:

$$	ext{Features} = [w_{	ext{mono1}}, \eta_{	ext{mono1}}, \gamma_{	ext{mono1}}] + [w_{	ext{mono2}}, \eta_{	ext{mono2}}, \gamma_{	ext{mono2}}] + [w_{	ext{PI}}, \eta_{	ext{PI}}, \gamma_{	ext{PI}}] + [w_{	ext{water/solv}}, \eta_{	ext{solv}}, \gamma_{	ext{solv}}]$$

where:
*   $w_i$ = weight fraction of component $i$ ($\sum w_i = 1.0$)
*   $\eta_i$ = pure component viscosity at jetting temperature (e.g., 40°C) [48]
*   $\gamma_i$ = pure component surface tension [48]

### B. Machine Learning Modeling (XGBoost/PyCaret)
Rather than relying solely on simplistic ideal mixture equations (like linear blending) which fail for non-ideal polar solvent-water interactions, we train an **XGBoost Regressor** [19, 30]:
1. **Viscosity Prediction ($\eta_{	ext{mix}}$)**: Predicts the dynamic viscosity of the final formulated blend at the target jetting temperature [48].
2. **Surface Tension Prediction ($\gamma_{	ext{mix}}$)**: Predicts the surface tension of the blend, which is highly non-linear due to surfactant-like behaviors of certain monomers.

### C. Calculating the Jetting Window (The Z-Number)
With the predicted $\eta_{	ext{mix}}$ and $\gamma_{	ext{mix}}$, your Python application will dynamically calculate the **Ohnesorge/Z-number** [48], which is the standard dimensionless parameter used in inkjet engineering to evaluate printability:

$$Z = rac{\sqrt{\gamma_{	ext{mix}} \cdot 
ho_{	ext{mix}} \cdot d}}{\eta_{	ext{mix}}}$$

where $d$ is the nozzle diameter of the printhead and $
ho_{	ext{mix}}$ is the density. 
*   **Printable Window**: The physics of drop-on-demand inkjet printing dictate that the ink is printable **only if $1 < Z < 10$**.
*   If $Z < 1$, viscous dissipation prevents droplet ejection.
*   If $Z > 10$, the droplet splits, forming unwanted satellite droplets that ruin print quality.

---

## 3. Model 2: Curing Performance & Photoinitiator Discovery

Once Model 1 ensures the ink can be jetted, Model 2 predicts if it will cure efficiently in its specific environment.

### A. Multi-Chemical Representation & Environment Input
As developed in v3, Model 2 takes the chemical fingerprints of the components (using **RDKit** Morgan Fingerprints [43]) and explicitly incorporates the solvent vehicle as numerical parameters:
1. **Monomer Fingerprint & Properties**: Average molecular weight, density of double bonds.
2. **Photoinitiator Fingerprint & Properties**: LogP (octanol-water partition coefficient, crucial to flag water solubility [43]), molar refractivity, and UV absorption maxima.
3. **Environmental Context**: A binary flag `Is_Aqueous` (1 for water-based, 0 for 100% UV-monomer/solvent) and the dielectric constant of the solvent vehicle.
4. **Process Parameters**: Weight percentage of PI ($\% \text{ PI}$) and the UV Lamp Energy Dose ($E_{	ext{UV}}$).

### B. Virtual Screening for Photoinitiator Discovery
To discover *new* or *modified* photoinitiators that work perfectly in specific environments (especially challenging aqueous-UV inks), we use **RDKit-driven Virtual Screening**:
1. **Anchor Library**: Extract known photoinitiator structures from PubChem/ChEMBL [47].
2. **Chemical Modifications (In-Silico)**: Apply virtual reactions (e.g., attaching water-solubilizing sulfonate, carboxylate, or PEG chains to a hydrophobic benzophenone or TPO core).
3. **Screening Evaluation**: Run these newly designed SMILES structures through Model 2. The model will calculate their LogP, check their compatibility with the aqueous environment (`Is_Aqueous = 1`), and predict the resulting $\% \text{ conversion}$. Only those with predicted conversion $>85\%$ are selected for potential synthesis.

---

## 4. Updated 5-Module Study Plan

This comprehensive roadmap perfectly aligns with your specialization requirements while delivering direct chemical and formulation value:

### Modulo 1: Cheminformatics & Feature Engineering (The Foundation)
*   **RDKit Essentials**: Working with SMILES, molecular visualization, and calculating chemical properties (LogP, molecular weight) [43, 44].
*   **Advanced Feature Engineering**: Creating mixture-based tabular records that merge weight fractions, pure solvent properties, and Morgan Fingerprints of monomers/PIs into one unified input vector.

### Modulo 2: Regression for Formulation and Curing (XGBoost & PyCaret)
*   **XGBoost Regressors**: Building separate models for $\eta_{	ext{mix}}$, $\gamma_{	ext{mix}}$, and $\% \text{ conversion}$ [19, 30].
*   **Model Validation**: Implementing custom train-test splits ensuring that the test set evaluates completely unseen monomer combinations or novel photoinitiator structures.

### Modulo 3: Computer Vision & Deep Learning (Specialization Core)
*   **Molecular Graphs (GNNs)**: Representing molecules as graphs to predict properties directly from chemical connectivity.
*   **CNN 2D Structure Recognition**: Generating 2D chemical structure drawings with RDKit and feeding them into a **Convolutional Neural Network (CNN)** (using Transfer Learning from Keras [4]) to classify photoinitiators into high/low reactivity classes. This fulfills your Deep Learning & Computer Vision curriculum requirements.

### Modulo 4: Multi-Objective Bayesian Optimization (Optuna)
*   Formulating the fitness function:
    $$	ext{Maximize } 	ext{Conversion}\% \quad 	ext{subject to} \quad 8 	ext{ cP} < \eta_{	ext{mix}} < 15 	ext{ cP} \quad 	ext{and} \quad 1 < Z < 10$$
*   Using Optuna to automatically suggest the ideal monomer/photoinitiator/water ratio.

### Modulo 5: Deployment & Containerization (MLOps)
*   **Docker**: Containerizing RDKit, PyTorch/TensorFlow, and XGBoost dependencies to ensure a conflict-free, reproducible environment.
*   **Streamlit Web Dashboard**: Building an interactive web app with sliders for formula composition (see details below).

---

## 5. Proposed Capstone Project: "Smart-Curing & Inkjet AI"

Here is the exact structure of the interactive web app you will present to your coach as your final deliverable:

1. **The Molecule Explorer (Discovery Side)**:
   * The user inputs a SMILES string of a novel photoinitiator (or selects a newly generated in-silico variant).
   * RDKit draws the molecule on-screen, computes its LogP [43], and Model 2 predicts its UV curing efficiency.
2. **The Formulation Simulator (Formulation Side)**:
   * Interactive **sliders** allow the user to define the percentage of Monomer A, Monomer B, Water/Solvent, and Photoinitiator.
   * **Model 1** instantly predicts the **mixture viscosity, surface tension, and calculates the Z-number (Ohnesorge)** to display a visual gauge: 🟢 **Printable**, 🟡 **Satellite Risk**, or 🔴 **Clogging/No Jetting**.
   * **Model 2** simultaneously displays a visual curing gauge predicting the **Double Bond Conversion %** and the predicted **glass transition temperature ($T_g$)** of the cured ink [43].
3. **The Formulation Auto-Optimizer**:
   * A button that triggers **Optuna** to find the absolute cheapest or most active formulation that satisfies both jetting and curing constraints.
