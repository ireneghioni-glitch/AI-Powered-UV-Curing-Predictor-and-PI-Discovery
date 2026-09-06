# QSAR Theory & Advanced Chemical Simulation of UV-Curing Conversion

This report outlines how to transition from a simplified heuristic target simulation to a scientifically rigorous, physics-informed **QSPR (Quantitative Structure-Property Relationship)** model for predicting UV-curing double-bond conversion (% Conversion) in inkjet and 100% reactive systems.

---

## 1. QSAR/QSPR Theory: Capturing Unsaturation and Double Bonds

In cheminformatics, Quantitative Structure-Property Relationship (QSPR) models do not treat "double bonds" as a single abstract number. Instead, they represent unsaturations, steric environments, and electronic configurations using specific, mathematically computed **molecular descriptors**.

To systematically detect and weight double bonds ($C=C$) using open-source tools like **RDKit**, several layers of descriptors are utilized:

### A. Substructure and Functional Group Counts (SMARTS Matching)
The most direct way to capture double bonds is by counting specific reactive substructures using SMARTS (Smiles Arbitrary Target Specification) patterns. In UV curing, not all double bonds are equal:
*   **Acrylates** ($\text{CH}_2=\text{CH}-\text{C}(=\text{O})-\text{O}-$): The standard reactive group, which polymerizes very quickly.
    *   *SMARTS Pattern*: `[CX3;H2]=[CX3;H1][CX3](=[OX1])[OX2]`
*   **Methacrylates** ($\text{CH}_2=\text{C}(\text{CH}_3)-\text{C}(=\text{O})-\text{O}-$): Sterically hindered by the $\alpha$-methyl group, reducing homopolymerization kinetics.
    *   *SMARTS Pattern*: `[CX3;H2]=[CX3]([CX4;H3])[CX3](=[OX1])[OX2]`
*   **Vinyl Ethers / Vinyl Esters**: Slower radical homopolymerization, often used in cationic curing or specialized hybrid systems.
    *   *SMARTS Pattern (Vinyl Ether)*: `[CX3;H2]=[CX3;H1][OX2]`

### B. Double Bond Density (Unsaturated Equivalent Weight)
In photopolymerization, the concentration of reactive sites per unit of mass is crucial. A small monomer like Acrylic Acid ($Mw \approx 72\text{ g/mol}$) has a much higher density of double bonds than a high-molecular-weight oligomer, even if both are monofunctional.
This is captured by calculating:
$$\text{Double Bond Density (DBD)} = \frac{\text{Number of } C=C \text{ double bonds}}{\text{Molecular Weight}} \quad (\text{mol/g})$$
This can be calculated programmatically in RDKit by dividing the SMARTS match count by `Descriptors.ExactMolWt`.

### C. Electrotopological State (E-State) Indices
Hall and Kier's E-State descriptors combine electronic and topological information. They quantify the electronic environment of specific atom types. For example:
*   `SdsCH` measures the electrotopological state of $=\text{CH}-$ groups in double bonds.
*   `Sd2C` measures the state of $=\text{C}<$ groups.
*   These indices allow an XGBoost or PyTorch model to learn how electron-withdrawing or electron-donating groups adjacent to the double bond alter its radical stability and propagation rate.

---

## 2. Incorporating the Chemical Environment (Solvent vs. Aqueous)

In inkjet UV formulations, the solvent environment (Water vs. Organic Solvent) dramatically affects the polymerization kinetics and final conversion due to several physical phenomena:

### A. Fotoinitiator (PI) Solubility and Partition Coefficient (LogP)
A photoinitiator must be molecularly dissolved to absorb UV light and generate radicals.
*   **In Organic Solvents**: Highly hydrophobic PIs (e.g., TPO, Omnirad 184) have high solubility, leading to efficient radical generation.
*   **In Aqueous Environments**: High-LogP (hydrophobic) PIs will aggregate or precipitate, completely shutting down curing. Water-soluble PIs (e.g., Irgacure 2959 or thioxanthone salts) have low LogP or ionic groups.
*   *QSAR Parameter*: `Descriptors.MolLogP` (from RDKit) is the key predictor here.

### B. Oxygen Inhibition and Solvent Viscosity
Radical photopolymerization is strongly inhibited by atmospheric oxygen, which scavenges active radicals to form inactive peroxy radicals ($\text{R}-\text{O}-\text{O}^\bullet$).
*   **The Solvent Effect**: Water has a relatively high oxygen solubility and low viscosity, which facilitates rapid oxygen diffusion into the curing film, delaying the onset of curing (induction period).
*   **Viscous Monomers**: Bulk or high-viscosity monomer environments limit oxygen diffusion, reducing oxygen inhibition and yielding higher conversion at lower UV energy doses.
*   *QSAR Parameter*: Initial formulation viscosity ($\eta$) and solvent diffusion coefficients.

---

## 3. Formulating a Chemically Realistic Target Simulation (Physics-Informed)

To generate synthetic target values ($\% \text{ Conversion}$) for your MVP dataset that mimic real photopolymerization thermodynamics and kinetics, you can implement a semi-empirical kinetic model.

The conversion of double bonds as a function of time ($t$) or UV energy dose ($E$) is commonly described by a pseudo-first-order kinetic equation with a saturation plateau:

$$\% \text{ Conversion} = \alpha_{\text{max}} \times \left(1 - e^{-k_{\text{eff}} \times E}\right)$$

Where:
*   $\alpha_{\text{max}}$ is the **maximum achievable conversion** (limited by vitrification/glass transition and oxygen inhibition).
*   $k_{\text{eff}}$ is the **effective rate constant** of polymerization.
*   $E$ is the **UV Energy Dose** ($\text{mJ/cm}^2$).

### A. Modeling $\alpha_{\text{max}}$ (Vitrification & Sterics)
As polymerization proceeds, the glass transition temperature ($T_g$) of the developing network rises. When the network $T_g$ exceeds the curing temperature ($T_{\text{cure}}$), the system vitrifies (turns to glass), freezing the mobility of unreacted double bonds and halting the reaction.
*   High-$T_g$ monomers (like Isobornyl Acrylate, IBOA, or Methyl Methacrylate, MMA) vitrify early, leaving significant unreacted monomer (lower $\alpha_{\text{max}}$).
*   Flexible, low-$T_g$ diacrylates (like PEGDA) maintain network mobility, allowing very high conversion ($\alpha_{\text{max}} \approx 90-95\%$).

We can model this as:
$$\alpha_{\text{max}} = 0.95 \times \left(1 - 0.002 \times \max(0, T_g^{\text{polymer}} - T_{\text{cure}})\right) \times \text{Steric Factor}$$

### B. Modeling $k_{\text{eff}}$ (PI Efficiency & Solubility)
The effective curing rate depends on radical generation and monomer reactivity:
$$k_{\text{eff}} = I_0 \times \Phi_{\text{PI}} \times \text{Solubility Factor} \times \text{Monomer Reactivity}$$

Where:
1.  **Monomer Reactivity**:
    *   Acrylates = $1.0$
    *   Methacrylates = $0.3$ (due to steric hindrance and lower propagation rate constants, $k_p$)
    *   Vinyl Ethers = $0.1$ (for radical systems)
2.  **Solubility Factor**:
    *   A function of the PI's LogP and the environment:
    *   If `Is_Aqueous == 1`:
        $$\text{Solubility Factor} = e^{-\sigma \times \max(0, \text{LogP} - 2.0)^2}$$ (penalizes hydrophobic PIs in water).
    *   If `Is_Aqueous == 0`:
        $$\text{Solubility Factor} = 1.0$$ (all commercial PIs dissolve well in organic monomers/solvents).

---

## 4. Python Implementation of the Chemical Simulation Engine

This script can be integrated directly into your Phase 3 pipeline to generate highly realistic, physics-informed synthetic targets:

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

def count_double_bonds(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    # SMARTS for aliphatic carbon-carbon double bonds
    patt = Chem.MolFromSmarts("[CX3]=[CX3]")
    return len(mol.GetSubstructMatches(patt))

def is_methacrylate(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    # SMARTS for methacrylate ester group
    patt = Chem.MolFromSmarts("[CX3;H2]=[CX3]([CX4;H3])[CX3](=[OX1])[OX2]")
    return mol.HasSubstructMatch(patt)

def simulate_chem_conversion(monomer_smiles, pi_smiles, is_aqueous, uv_dose, t_cure=25):
    # 1. Compute RDKit Descriptors
    mono_mw = Descriptors.ExactMolWt(Chem.MolFromSmiles(monomer_smiles))
    pi_logp = Descriptors.MolLogP(Chem.MolFromSmiles(pi_smiles))
    num_cc = count_double_bonds(monomer_smiles)
    
    if num_cc == 0:
        return 0.0 # No reactive double bonds
        
    # 2. Compute Double Bond Density (DBD)
    db_density = num_cc / mono_mw # mol/g
    
    # 3. Assess Monomer Reactivity Class
    if is_methacrylate(monomer_smiles):
        monomer_reactivity = 0.35 # Slow kinetics due to alpha-methyl group
        tg_est = 105.0 # Typical PMMA/methacrylate homopolymer Tg in °C
    else:
        monomer_reactivity = 1.0 # Fast acrylate kinetics
        tg_est = -20.0 if "PEG" in monomer_smiles else 50.0 # Flexible vs rigid
        
    # 4. Assess Photoinitiator Solubility in Environment
    if is_aqueous:
        # Penalize highly hydrophobic photoinitiators (high LogP) in water
        # Optimal LogP for water is <= 2.0 (e.g., Irgacure 2959 has LogP ~ 0.8)
        solubility_factor = np.exp(-0.8 * max(0, pi_logp - 2.0)**2)
    else:
        solubility_factor = 1.0 # High solubility in organic veichles
        
    # 5. Vitrification Effect (Tg vs Curing Temperature)
    # Early vitrification limits the maximum conversion
    if tg_est > t_cure:
        alpha_max = 0.95 - 0.0015 * (tg_est - t_cure)
    else:
        alpha_max = 0.95 # Flexible network allows near-complete conversion
        
    # 6. Polymerization Rate Constant (k_eff)
    # Combines monomer kinetics, PI solubility, and double bond concentration
    k_eff = 0.015 * monomer_reactivity * solubility_factor * (1.0 + db_density * 50)
    
    # 7. Final Conversion Kinetic Equation
    conversion = alpha_max * (1 - np.exp(-k_eff * uv_dose)) * 100
    
    # Add minor Gaussian noise to simulate experimental error (std=1.5%)
    noise = np.random.normal(0, 1.5)
    final_conversion = conversion + noise
    
    return min(max(final_conversion, 0.0), 100.0)
```

By leveraging this physics-informed script, your dataset will show highly realistic trends:
1.  **Acrylate monomers** will cure significantly faster than **methacrylate monomers**.
2.  **Hydrophobic photoinitiators** (like TPO) will show near-zero curing in water, but excellent curing in solvent, while hydrophilic initiators will remain highly efficient in both.
3.  The **UV Dose curves** will exhibit a realistic exponential saturation plateau matching standard photo-DSC and real-time FTIR curing data.
