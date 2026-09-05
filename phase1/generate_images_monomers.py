'''
What does this module do?
=========================

It takes a SMILES string (e.g., "C=C(C)C(=O)OCC(COC(=O)C(=C)C)(COC(=O)C(=C)C)COC(=O)C(=C)C" for TMPTA)
and converts it into a fixed-size, grayscale 2D image (e.g., 224x224 pixels).

The image is a drawing of the molecular structure, with atoms and bonds
represented graphically.
'''

import pandas as pd
import numpy as np
import cv2
from rdkit import Chem
from rdkit.Chem import Draw
from pathlib import Path


# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"

INPUT_CSV = DATA_DIR / "molecules_monomers.csv"

# Compressed NumPy array of all images (shape: N, 224, 224)
OUTPUT_NPZ = IMAGES_DIR / "molecular_images_monomers.npz"
# Metadata: name, smiles, role, augment
OUTPUT_META = DATA_DIR / "molecular_metadata_monomers.csv"
IMG_SIZE = (224, 224)  # Standard for most pre-trained CNNs (ResNet, MobileNet).
PREVIEW = DATA_DIR / "preview_monomers.png"


# ==================== FUNCTIONS ====================

# get grey image of molecules
def smiles_to_greyscale(smiles):
    '''
    Generate 2D in gray scale image of molecule from SMILES.
    Pattern recognition (bonds, rings) does not require color,
    so we use less memory by doing this.'''
    # transform smiles string into molecule obj
    mol = Chem.MolFromSmiles(smiles)
    # in case it fails, raise of ValueError
    if mol is None:
        raise ValueError(f'Invalid SMILES: {smiles}')
    # draw 2D structure (RGB with 3 channels by default)
    img = Draw.MolToImage(mol, size=IMG_SIZE)
    # reduces image at 1 channel (gray scale)
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return gray


# rotate molecule
def augment_rotations(image):
    '''
    It takes an input image and rotates it around the
    center by 90, 180, and 270 degrees.

    To increase the variety of training examples and make the CNN invariant to rotation,
    we apply rotations of 90°, 180°, and 270°.
    For molecules, rotating the drawing does not change the chemical structure;
    it only changes the orientation in the image.

    Returns a list of 3 images.'''
    h, w = image.shape
    # Divide the dimensions by 2, discarding the decimals, to get center coordinates.
    center = (w // 2, h // 2)
    # In image processing, the array dimensions are expressed as (height, width),
    # whereas screen coordinates use the standard Cartesian system (x, y) - that is, (width, height).
    # For this reason, w is placed first.
    rotated = []
    for angle in [90, 180, 270]:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        rotated.append(rot)
    return rotated


# process one row of CSV (image + rotations)
def process_molecule_row(row):
    name = row['name']
    smiles = row['smiles']
    role = row['role']

    # Check if the SMILES exists
    if pd.isna(smiles) or smiles is None:
        print(f'    [FAIL] Skipping {name}: no SMILES')
        return []

    # Call smiles_to_grayscale() for the original image
    try:
        orig = smiles_to_greyscale(smiles)
    except Exception as e:
        print(f'    [ERROR] Error processing {name}: {e}')
        return []

    # For each rotation, it creates a record with an 'augment' flag
    # indicating whether it is original or rotated.
    records = [
        {
            'image': orig,
            'name': name,
            'smiles': smiles,
            'role': role,
            'augment': 'orig'
        }
    ]
    for idx, rot in enumerate(augment_rotations(orig)):
        records.append({
            'image': rot,
            'name': name,
            'smiles': smiles,
            'role': role,
            'augment': f'rot{90 * (idx + 1)}'
        })
    return records


def show_preview(images, meta, n=4):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, n, figsize=(12, 3))
    for i in range(n):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f"{meta.iloc[i]['name']} {meta.iloc[i]['augment']}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(PREVIEW, dpi=100)
    print(f"   Preview saved to {PREVIEW}")


# main function
def main():
    '''
    Orchestrator.'''
    print(f'Loading molecules CSV from {INPUT_CSV}')
    # read CSV
    df = pd.read_csv(INPUT_CSV)
    print(f'Found {len(df)} molecules.')

    # Process each row by calling process_molecule_row
    # and accumulate the results in all_data.
    all_data = []
    for idx, row in df.iterrows():
        print(f'    [{idx + 1}/{len(df)}] Processing {row["name"]}')
        all_data.extend(process_molecule_row(row))

    # Extracts the images as NumPy arrays and the metadata as a DataFrame.
    images = np.array([rec['image'] for rec in all_data], dtype=np.uint8)
    meta = pd.DataFrame([{k: v for k, v in rec.items() if k != 'image'} for rec in all_data])

    # Save everything using `np.savez_compressed` (it takes up little space)
    # along with the metadata CSV.
    np.savez_compressed(OUTPUT_NPZ, images=images)
    meta.to_csv(OUTPUT_META, index=False)

    if len(images) >= 4:
        show_preview(images, meta)

    print(f'\nGenerated {len(images)} images.')
    print(f'    saved to {OUTPUT_NPZ}')
    print(f'    Metadata saved to {OUTPUT_META}')


if __name__ == "__main__":
    main()