'''
Batch image generator for photoinitiator molecules.

Reads ``data/molecules_PIs.csv`` and produces:
  - ``images/molecular_images.npz``  : array of shape (N, 224, 224), uint8
  - ``images/molecular_metadata.csv``: name, SMILES, role, augment per image
  - ``data/preview.png``             : visual sanity check of the first 4 images

IMPORTANT!
Re-run this script every time ``molecules_PIs.csv`` is updated.

Run from the PROJECT ROOT:

    python -m phase1.generate_images_PIs

The actual image-transformation logic (SMILES → grayscale, rotations,
row processing, preview) lives in ``shared/molecule_images.py`` and is
shared with the monomer generator and the Phase 5 inference pipeline.
'''

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from shared.molecule_images import process_molecule_row, show_preview


# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = DATA_DIR / "molecules_PIs.csv"
OUTPUT_NPZ = IMAGES_DIR / "molecular_images.npz"
OUTPUT_META = IMAGES_DIR / "molecular_metadata.csv"
PREVIEW_PATH = DATA_DIR / "preview.png"


# ==================== MAIN EXECUTION ====================

def main():
    '''
    Batch entry point: convert every SMILES in ``INPUT_CSV`` to images.

    Workflow
    --------
    1. Read ``data/molecules_PIs.csv`` (columns: name, smiles, role).
    2. For each row, call ``process_molecule_row`` (from the shared module)
       to produce 4 records (original + 90° + 180° + 270° rotations).
    3. Stack all images into a NumPy array of shape (N, 224, 224).
    4. Save the array to ``images/molecular_images.npz`` (compressed).
    5. Save the per-image metadata to ``images/molecular_metadata.csv``.
    6. Save a preview of the first 4 images to ``data/preview.png``.

    Reads
    -----
    - ``data/molecules_PIs.csv``

    Writes
    ------
    - ``images/molecular_images.npz``
    - ``images/molecular_metadata.csv``
    - ``data/preview.png``

    Notes
    -----
    Must be run from the project root:

        python -m phase1.generate_images_PIs
    '''
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
    meta = pd.DataFrame(
        [{k: v for k, v in rec.items() if k!='image'} for rec in all_data]
    )

    # Save everything using `np.savez_compressed` (it takes up little space) 
    # along with the metadata CSV.
    np.savez_compressed(OUTPUT_NPZ, images=images)
    meta.to_csv(OUTPUT_META, index=False)

    if len(images) >= 4:
        show_preview(images, meta, PREVIEW_PATH)

    print(f'\nGenerated {len(images)} images.')
    print(f'    saved to {OUTPUT_NPZ}')
    print(f'    Metadata saved to {OUTPUT_META}')


if __name__ == "__main__":
    main()
