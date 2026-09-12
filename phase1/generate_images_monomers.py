'''
Batch image generator for monomer molecules.

Reads ``data/molecules_monomers.csv`` and produces the corresponding
NPZ + metadata + preview files.

IMPORTANT!
Re-run this script every time ``molecules_monomers.csv`` is updated.

Run from the PROJECT ROOT:

    python -m phase1.generate_images_monomers

The actual image-transformation logic lives in ``shared/molecule_images.py``.
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

INPUT_CSV = DATA_DIR / "molecules_monomers.csv"
OUTPUT_NPZ = IMAGES_DIR / "molecular_images_monomers.npz"
OUTPUT_META = IMAGES_DIR / "molecular_metadata_monomers.csv"
PREVIEW_PATH = DATA_DIR / "preview_monomers.png"


# ==================== MAIN EXECUTION ====================

def main():
    '''
    Batch entry point: convert every monomer SMILES in ``INPUT_CSV`` to images.

    Same workflow as ``phase1.generate_images_PIs.main``; the only
    differences are the input/output paths and the family-specific
    data source.
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
        [{k: v for k, v in rec.items() if k != 'image'} for rec in all_data]
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