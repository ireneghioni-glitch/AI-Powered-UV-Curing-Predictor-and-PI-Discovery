# Phase 1: Data Ingestion and Chemical Preprocessing – Detailed Step‑by‑Step Guide
This guide walks you through the first phase of your MVP pipeline. By the end of this phase you will have a curated dataset of molecular images (grayscale, fixed size) ready for the Computer Vision feature extraction (Phase 2).

You will learn how to:
* Set up a Python environment with all necessary libraries.
* btain SMILES strings for monomers and photoinitiators from public databases.
* Convert SMILES to 2D chemical diagrams using RDKit.
* Apply standard image preprocessing (grayscale conversion, resizing).
* Perform data augmentation (rotations) to artificially enlarge your dataset.
* Organise and save the processed images for efficient loading later.

## 1. Environment Setup
Create a dedicated Python environment (conda or venv) and install the required packages:

### 1. Create and activate a conda environment (recommended)
```bash
conda create -n chemvision python=3.9
conda activate chemvision
```

### 2. Install core libraries
```bash
pip install rdkit-pypi opencv-python numpy matplotlib pandas pillow
```

### Why these libraries?

* **RDKit** – handles chemical SMILES and generates 2D molecular drawings.
* **OpenCV** – performs image preprocessing (grayscale, resizing, rotations).
* **NumPy** – stores image data as arrays.
* **Matplotlib** – for visualisation (checking your images).
* **Pandas** – to manage the metadata (SMILES, properties) in CSV format.

## 2. Obtain a List of Molecules (SMILES Strings)
You need a collection of monomers and photoinitiators. For the MVP, you can start with a small set of well‑known commercial compounds. Later you can scale up using public databases.

### Option A – Manual entries (for immediate testing)
Create a Python dictionary or CSV with at least 10–20 molecules:

| Role | Name | SMILES |
| --- | --- | --- |
| Monomer | Acrylic acid | C=CC(=O)O |
| Monomer | Methyl methacrylate | CC(=C)C(=O)OC |
| Monomer | Styrene | C=Cc1ccccc1 |
| Photoinitiator | Benzophenone | O=C(c1ccccc1)c2ccccc2 |
| Photoinitiator | TPO (diphenyl) | CC(C)(C)P(=O)(c1ccccc1)c2ccccc2 |
| Photoinitiator | Irgacure 184 | CC(=O)c1ccccc1 (or check exact) |

Save them in a CSV file (`molecules.csv`) with columns: `name`, `smiles`, `role` (monomer/PI).

### Option B – Programmatic fetching from PubChem (using PubChemPy)
If you want to automatically retrieve SMILES for a given name, install pubchempy and write a script:

```bash
pip install pubchempy
```

```python
import pubchempy as pcp

def get_smiles_by_name(name):
    try:
        compound = pcp.get_compounds(name, 'name')[0]
        return compound.isomeric_smiles
    except IndexError:
        return None

# Example
print(get_smiles_by_name("Benzophenone"))  # O=C(c1ccccc1)c2ccccc2
```

You can create a list of target names and fetch their SMILES – but be careful with rate limits and always cache results locally.

## 3. Convert SMILES to Grayscale Images
The core function that transforms a SMILES string into a fixed‑size grayscale image.

```python
from rdkit import Chem
from rdkit.Chem import Draw
import cv2
import numpy as np

def smiles_to_grayscale(smiles, size=(224, 224)):
    """
    Convert a SMILES string to a grayscale image array (numpy uint8).
    Args:
        smiles (str): valid SMILES
        size (tuple): desired (width, height)
    Returns:
        np.ndarray: grayscale image, shape (height, width)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Generate a 2D drawing (RDKit returns a PIL Image)
    img = Draw.MolToImage(mol, size=size)
    # Convert PIL Image to numpy array (RGB)
    img_np = np.array(img)
    # Convert RGB -> grayscale using OpenCV
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return gray
```
**Test it**:

```python
import matplotlib.pyplot as plt

smiles = "O=C(c1ccccc1)c2ccccc2"  # benzophenone
gray_img = smiles_to_grayscale(smiles)
plt.imshow(gray_img, cmap='gray')
plt.title("Benzophenone")
plt.axis('off')
plt.show()
```
You should see a clear black‑and‑white structural formula.

## 4. Data Augmentation (Rotations)
To increase the variety of training examples and make the CNN invariant to rotation, you can apply rotations of 90°, 180°, and 270°. For molecules, rotating the drawing does not change the chemical structure; it only changes the orientation in the image.

Implementation using **OpenCV**:

```python
def augment_rotations(image):
    """
    Generate three rotated versions of the input grayscale image.
    Returns a list of images: [90°, 180°, 270°] (clockwise).
    """
    h, w = image.shape
    center = (w // 2, h // 2)
    rotated = []
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        rotated.append(rot)
    return rotated
```
> [!NOTE] 
> You may also consider horizontal/vertical flips (mirroring). However, flips can invert stereochemistry if the molecule has chiral centres. For simple acrylates and benzophenones (no chirality), flips are safe, but we recommend sticking to rotations to avoid any conceptual ambiguity.

**Usage**:

```python
original = smiles_to_grayscale("C=CC(=O)O")
rotations = augment_rotations(original)
# Now you have 4 images: original + 3 rotations
```

## 5. Build the Full Dataset
Now you will loop over all molecules in your CSV, generate the original grayscale image and its rotations, and store them in a structured way.

### a. Define a function to process one molecule and return augmented images with metadata
```python
import pandas as pd

def process_molecule_row(row, size=(224, 224)):
    """
    Given a DataFrame row with 'smiles' and 'role', generate images.
    Returns a list of dicts, each containing the image array and metadata.
    """
    smiles = row['smiles']
    name = row['name']
    role = row['role']  # 'monomer' or 'PI'
    
    try:
        orig = smiles_to_grayscale(smiles, size=size)
    except Exception as e:
        print(f"Error processing {name}: {e}")
        return []
    
    # Original image
    records = [{'image': orig, 'name': name, 'smiles': smiles, 'role': role, 'augment': 'orig'}]
    
    # Rotations
    for idx, rot in enumerate(augment_rotations(orig)):
        records.append({
            'image': rot,
            'name': name,
            'smiles': smiles,
            'role': role,
            'augment': f'rot{90*(idx+1)}'
        })
    return records
```
### b. Load your CSV and iterate
```python
df = pd.read_csv('molecules.csv')
all_data = []

for _, row in df.iterrows():
    all_data.extend(process_molecule_row(row))

print(f"Generated {len(all_data)} images from {len(df)} molecules.")
```
### c. Save images and metadata
You have two common options:

1. **Save as numpy arrays** (`.npy`) – efficient for loading in deep learning.

2. **Save as image files** (`.png`) – easier to inspect with any viewer.

We recommend saving as a single `.npz` file (compressed) together with a metadata CSV.

```python
# Prepare arrays and metadata lists
images = np.array([rec['image'] for rec in all_data], dtype=np.uint8)
meta = pd.DataFrame([{k: v for k, v in rec.items() if k != 'image'} for rec in all_data])

# Save
np.savez_compressed('molecular_images.npz', images=images)
meta.to_csv('molecular_metadata.csv', index=False)
```
Now you have a compact dataset ready for Phase 2.

## 6. Visual Quality Check
Always verify a few samples to ensure the preprocessing is correct.

```python
import matplotlib.pyplot as plt

# Load back the npz to check
data = np.load('molecular_images.npz')
imgs = data['images']
meta = pd.read_csv('molecular_metadata.csv')

fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < len(imgs):
        ax.imshow(imgs[i], cmap='gray')
        ax.set_title(meta.iloc[i]['name'] + ' ' + meta.iloc[i]['augment'])
        ax.axis('off')
plt.tight_layout()
plt.show()
```

## 7. Optional: Data Augmentation with More Transforms
If you want to further increase dataset size, you can add:

* **Small random shifts** (translation).
* **Gaussian noise**.
* **Brightness/contrast adjustments** (though molecular drawings are clean, so less needed).

However, for the MVP, rotations are sufficient. You can always expand later.

## 8. Summary of Phase 1 Deliverables
After completing this phase, your project folder should contain:

* `molecules.csv` – the original list of SMILES.
* `molecular_images.npz` – compressed NumPy array of all images (original + augmentations).
* `molecular_metadata.csv` – corresponding metadata (name, SMILES, role, augmentation type).

You have also written the functions `smiles_to_grayscale()` and `augment_rotations()` that can be reused in later phases (e.g., during deployment, when a user enters a new SMILES).

## 9. Next Steps (Preview of Phase 2)
Now that you have a dataset of grayscale molecular images, you will:

* Load the images into a PyTorch or Keras `Dataset`/`DataLoader`.
* Use a pre‑trained CNN (e.g., ResNet50 or MobileNetV2) as a feature extractor, freezing its weights.
* Pass each image through the CNN and extract the penultimate layer’s output (a dense vector) – this becomes the “visual embedding” of the molecule.

We’ll cover that in the next detailed guide.

## 10. Troubleshooting & Tips
* **RDKit drawing quality**: Sometimes RDKit may produce slightly different orientations for the same molecule. That’s fine – the CNN will learn rotation invariance from your augmentation.
* **Memory**: If you have thousands of molecules, storing all images in memory at once might be heavy. You can generate images on‑the‑fly during training (by converting SMILES to images in the __getitem__ method). For the MVP, storing them is simpler and faster.
* **SMILES validity**: Always check Chem.MolFromSmiles() returns None – handle errors gracefully.
* **Channel dimension**: Grayscale images have shape (224, 224). Most CNNs expect a channel dimension, so you will need to add an extra axis: np.expand_dims(img, axis=-1) or use np.stack to create (N, 224, 224, 1).

Now you are ready to start Phase 1.
> [!NOTE]  
> Important: Commit your code and data (except large arrays) to your GitHub repository as soon as you have a working version.

