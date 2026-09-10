'''
Shared molecular image utilities: SMILES → grayscale image, data
augmentation, batch row processing, and visual preview.

Used by:
- phase1/generate_images_PIs.py      (batch, photoinitiators)
- phase1/generate_images_monomers.py (batch, monomers)
- inference/pipeline.py              (runtime, single-molecule inference)

Why this module exists
----------------------
The SMILES → grayscale transformation is a *contract* between training
time (Phase 1) and inference time (Phase 5). If the two pipelines
preprocess slightly differently (different image size, different
grayscale formula, different padding) the embeddings fed to the
classifier will be off-distribution and predictions will silently
degrade with no error message. 
Centralizing the logic here makes that impossible: there is only one 
implementation, imported everywhere.

No class is used because the functions are pure (no state). This is the
same design philosophy as the module-level helpers in pubchem_client.py.
'''

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


# ==================== CONSTANTS ====================

IMG_SIZE = (224, 224)


# ==================== CORE TRANSFORMATIONS ====================

def smiles_to_grayscale(smiles: str, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    '''
    Convert a SMILES string to a grayscale image (uint8, shape H×W).

    Pipeline
    --------
    1. SMILES → RDKit molecule object (validates the SMILES).
    2. Molecule → PIL image (RDKit 2D drawing).
    3. PIL image → NumPy array (H, W, 3).
    4. OpenCV: RGB → grayscale (H, W).

    Parameters
    ----------
    smiles : str
        Valid SMILES string.
    size : (int, int), default IMG_SIZE
        Output image dimensions. Do not change this unless the CNN
        was retrained with a different input size.

    Returns
    -------
    np.ndarray
        Grayscale image, dtype uint8, shape (height, width).

    Raises
    ------
    ValueError
        If RDKit cannot parse the SMILES.
    '''
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    img = Draw.MolToImage(mol, size=size)
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return gray


def augment_rotations(image: np.ndarray) -> list[np.ndarray]:
    '''
    Return three rotated copies of the input image: 90°, 180°, 270°.

    Rotation is a physically meaningful augmentation for 2D molecular
    drawings: the molecule is orientation-independent in the plane, but
    the CNN must be trained to ignore that. 
    Rotations preserve the chemistry (unlike horizontal/vertical flips, 
    which can invert stereochemistry annotations).
    '''
    h, w = image.shape
    center = (w // 2, h // 2)
    rotated = []
    for angle in (90, 180, 270):
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
        rotated.append(rot)
    return rotated


# ==================== BATCH HELPERS ====================

def process_molecule_row(row: pd.Series) -> list[dict]:
    '''
    Process a single row of a molecules CSV into four image records.

    Input row schema (both PI and monomer CSVs): columns ``name``,
    ``smiles``, ``role``.

    Returns a list of 0 or 4 records:
    - 0 records if SMILES is missing/invalid (the row is skipped with a
      diagnostic print).
    - 4 records otherwise: one original + three rotated variants.

    Each record is a dict with keys:
        image    : np.ndarray (grayscale, 224×224, uint8)
        name     : str
        smiles   : str
        role     : str
        augment  : 'orig' | 'rot90' | 'rot180' | 'rot270'
    '''
    name = row["name"]
    smiles = row["smiles"]
    role = row["role"]

    if pd.isna(smiles) or smiles is None:
        print(f"  [FAIL] Skipping {name}: no SMILES")
        return []

    try:
        orig = smiles_to_grayscale(smiles)
    except Exception as exc:
        print(f"  [ERROR] Error processing {name}: {exc}")
        return []

    records = [{
        "image": orig,
        "name": name,
        "smiles": smiles,
        "role": role,
        "augment": "orig",
    }]

    for idx, rot in enumerate(augment_rotations(orig)):
        records.append({
            "image": rot,
            "name": name,
            "smiles": smiles,
            "role": role,
            "augment": f"rot{90 * (idx + 1)}",
        })

    return records


# ==================== VISUALIZATION ====================

def show_preview(
    images: np.ndarray,
    meta: pd.DataFrame,
    output_path: Path,
    n: int = 4,
) -> None:
    '''
    Save a horizontal preview of the first ``n`` images to ``output_path``.

    Used as a visual sanity check after image generation. Parameterized
    on ``output_path`` because the destination is family-specific
    (PI vs monomers) and cannot be a module-level constant here.
    '''
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, n, figsize=(12, 3))
    for i in range(n):
        axes[i].imshow(images[i], cmap="gray")
        axes[i].set_title(f"{meta.iloc[i]['name']} {meta.iloc[i]['augment']}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)   # prevent figure accumulation in long-running processes
    print(f"   Preview saved to {output_path}")