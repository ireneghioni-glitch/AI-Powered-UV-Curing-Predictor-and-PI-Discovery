'''
Batch resolver for monomer SMILES.

IMPORTANT!
Re-run this script every time ``molecules_config_monomers`` is updated,
so that ``data/molecules_monomers.csv`` stays in sync with the curated
list.

Run from the PROJECT ROOT (not from inside phase1/):

    python -m phase1.fetch_molecules_monomers

The actual PubChem resolution logic (REST calls, cache, manual fallback,
safe CSV I/O) lives in ``shared/pubchem_client.py``.
'''


from __future__ import annotations

import pandas as pd
from pathlib import Path

from shared.pubchem_client import PubChemClient


# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MONO_CSV = DATA_DIR / "molecules_monomers.csv"
CACHE_FILE = DATA_DIR / "smiles_cache_monomers.csv"


# ==================== MANUAL FALLBACK DICTIONARY (monomers only) ====================
MANUAL_SMILES_MONO = {
        "TMPTA": "C=C(C)C(=O)OCC(COC(=O)C(=C)C)(COC(=O)C(=C)C)COC(=O)C(=C)C",
        "DEGDA": "C=C(C)C(=O)OCCOCCOC(=O)C(=C)C",
        "HDDA": "C=CC(=O)OCCCCCCOC(=O)C=C",
        "PEGDA": "C=C(C)C(=O)OCCOCCOCCOC(=O)C(=C)C",
        "HEMA": "C=C(C)C(=O)OCCO",
        "MMA": "C=C(C)C(=O)OC",
        "Butyl acrylate": "C=CC(=O)OCCCC",
        "Acrylic acid": "C=CC(=O)O",
        "Styrene": "C=CC1=CC=CC=C1",
        "IBOA": "C=CC(=O)OC1C2CC(C2)C1(C)C",
    }


# ==================== CLIENT INSTANCE ====================
# A single PubChemClient instance for the monomer family.
# Phase 5 imports this object directly and calls
# ``MONO_CLIENT.fetch_single_molecule(name, role="monomer")``.

MONO_CLIENT = PubChemClient(
    manual_smiles=MANUAL_SMILES_MONO,
    cache_file=CACHE_FILE,
    output_csv=OUTPUT_MONO_CSV,
)


# ==================== MOLECULES LIST ====================

'''
IMPORTANT!
When adding new monomers, specify trade name(s) as `primary_names` and
insert value None as `alt_name`, or viceversa.
'''

molecules_config_monomers = [
    # ---------- MONOMERS ----------
    {"primary_names": ["TMPTA"], "alt_name": "Trimethylolpropane triacrylate", "role": "monomer"},
    {"primary_names": ["DEGDA"], "alt_name": "Diethylene glycol diacrylate", "role": "monomer"},
    {"primary_names": ["HDDA"], "alt_name": "1,6-Hexanediol diacrylate", "role": "monomer"},
    {"primary_names": ["PEGDA"], "alt_name": "Poly(ethylene glycol) diacrylate", "role": "monomer"},
    {"primary_names": ["HEMA"], "alt_name": "2-Hydroxyethyl methacrylate", "role": "monomer"},
    {"primary_names": ["MMA"], "alt_name": "Methyl methacrylate", "role": "monomer"},
    {"primary_names": ["Butyl acrylate"], "alt_name": "Butyl acrylate", "role": "monomer"},
    {"primary_names": ["Acrylic acid"], "alt_name": "Acrylic acid", "role": "monomer"},
    {"primary_names": ["Styrene"], "alt_name": "Styrene", "role": "monomer"},
    {"primary_names": ["IBOA"], "alt_name": "Isobornyl acrylate", "role": "monomer"},
]


# ==================== BACKWARD-COMPAT WRAPPER ====================
def fetch_single_monomer(name: str) -> str | None:
    '''
    Legacy thin wrapper.
    Returns only the SMILES string (or ``None``).

    Kept for backward compatibility with any Phase 1 code that still
    imports it.
    New code (Phase 5) should call
    ``MONO_CLIENT.fetch_single_molecule(name, role="monomer")`` directly
    to get the full ``{"name", "smiles", "role"}`` dict.
    '''
    result = MONO_CLIENT.fetch_single_molecule(name, role="monomer")
    return result["smiles"] if result else None


# ==================== MAIN EXECUTION ====================
def main():
    '''
    Batch entry point: resolve the canonical SMILES for every monomer
    declared in ``molecules_config_monomers`` and persist the results to
    ``data/molecules_monomers.csv``.

    Workflow
    --------
    1. Iterate over ``molecules_config_monomers`` (list of dicts, each
       with ``primary_names``, optional ``alt_name`` and ``role``).
    2. For each entry, call :meth:`PubChemClient.get_smiles_robust`,
       which resolves the SMILES using, in order of preference:
         (a) the local cache ``data/smiles_cache_monomers.csv``,
         (b) the manual fallback dictionary ``MANUAL_SMILES``,
         (c) the PubChem PUG REST API (name -> CID -> CanonicalSMILES).
    3. Collect one record per molecule: ``{"name", "smiles", "role"}``.
    4. Overwrite ``data/molecules_monomers.csv`` (fully rewritten every
       run, so it always reflects the current configuration).
    5. Print a summary (found / total, list of missing molecules) and a
       preview of the first 10 rows.

    Reads
    -----
    - Module constant ``molecules_config_monomers``.
    - ``data/smiles_cache_monomers.csv`` (via ``MONO_CLIENT``).

    Writes
    ------
    - ``data/molecules_monomers.csv``     — final dataset consumed by
      Phases 2–5.
    - ``data/smiles_cache_monomers.csv``  — updated on every successful
      lookup.

    Notes
    -----
    Must be run from the project root:
        ``python -m phase1.fetch_molecules_monomers``
    Re-run every time ``molecules_config_monomers`` changes.
    '''
    print('Starting fetching SMILES using PubChem REST API + manual fallback...')
    print(f'Total molecules to process: {len(molecules_config_monomers)}\n')

    results = []
    for idx, item in enumerate(molecules_config_monomers, 1):
        primary_names = item["primary_names"]
        alt = item["alt_name"]
        role = item["role"]

        print(f'[{idx}/{len(molecules_config_monomers)}] Processing: {", ".join(primary_names)} (role: {role})')

        smiles, used_name = MONO_CLIENT.get_smiles_robust(primary_names, alt)
        if smiles is not None:
            print(f'    DEBUG: smiles = {smiles[:60]}...')
        else:
            print(f'    DEBUG: smiles = None')
        results.append({
            "name": used_name if used_name else primary_names[0],
            "smiles": smiles,
            "role": role
        })

        print("---")

    df_final = pd.DataFrame(results)
    df_final.to_csv(OUTPUT_MONO_CSV, index=False)

    found = df_final['smiles'].notna().sum()
    total = len(df_final)
    print(f'\n[COMPLETED] CSV saved as "{OUTPUT_MONO_CSV}"')
    print(f'    Found molecules: {found} over {total}')
    if found < total:
        missing = df_final[df_final['smiles'].isna()]['name'].tolist()
        print(f'    Not found: {missing}')

    print('\nPreview of generated CSV:')
    print(df_final.head(10).to_string())


if __name__ == "__main__":
    main()