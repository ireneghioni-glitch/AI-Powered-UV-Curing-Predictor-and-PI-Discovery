'''
Batch resolver for photoinitiator SMILES.

IMPORTANT!
Re-run this script every time ``molecules_config`` is updated, so that
``data/molecules_PIs.csv`` stays in sync with the curated list.

Run from the PROJECT ROOT (not from inside phase1/):

    python -m phase1.fetch_molecules_PIs

The actual PubChem resolution logic (REST calls, cache, manual fallback,
safe CSV I/O) lives in ``shared/pubchem_client.py``.

==================== FUTURE REFACTOR: EXTERNALIZE MOLECULE CONFIG ====================
Currently ``molecules_config`` is hardcoded: adding a molecule requires
editing this file.

Planned refactor: move the list to ``data/molecules_config.json``, load it
via a ``config_loader.py``, and expose ``PI_CLIENT.fetch_single_molecule``
so Phase 5 can add new molecules at runtime
(check CSV -> add to JSON -> fetch -> update CSV).

Details: see ``phase1/docs/future_refactor_externalize_config.md``.
=====================================================================================
'''

from __future__ import annotations

import pandas as pd
from pathlib import Path

from shared.pubchem_client import PubChemClient


# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PI_CSV = DATA_DIR / "molecules_PIs.csv"
CACHE_FILE = DATA_DIR / "smiles_cache_PIs.csv"


# manual fallback dictionary moved here from get_smiles_robust()
# ==================== MANUAL FALLBACK DICTIONARY ====================
MANUAL_SMILES = {
    "Irgacure 651": "COC(OC)(C1=CC=CC=C1)C(=O)C2=CC=CC=C2",
    "Darocur 1173": "CC(C)(O)C(=O)C1=CC=CC=C1",
    "Irgacure 184": "OC1(CCCCC1)C(=O)C2=CC=CC=C2",
    "Irgacure 369": "CN(C)C(CC1=CC=CC=C1)(C(=O)C2=CC=CC=C2)N3CCOCC3",
    "Irgacure 907": "CC1=CC=C(C=C1)SC(C)(C)C(=O)C2=CC=CC=C2",
    "TPO": "CC(C)(C)P(=O)(C1=CC=CC=C1)C(=O)C2=CC=CC=C2",
    "TPO-L": "CCOP(=O)(C1=CC=CC=C1)C(=O)C2=CC=CC=C2",
    "Irgacure 819": "CC(C)(C)P(=O)(C(=O)C1=CC=CC=C1)C(=O)C2=CC=CC=C2",
    "Benzophenone": "O=C(C1=CC=CC=C1)C2=CC=CC=C2",
    "4-Methylbenzophenone": "CC1=CC=C(C=C1)C(=O)C2=CC=CC=C2",
    "4-Phenylbenzophenone": "C1=CC=C(C=C1)C2=CC=C(C=C2)C(=O)C3=CC=CC=C3",
    "Methyl o-benzoylbenzoate": "COC(=O)C1=CC=CC=C1C(=O)C2=CC=CC=C2",
    "ITX": "CC(C)C1=CC2=C(C=C1)C(=O)C3=CC=CC=C3S2",
    "Thioxanthone": "O=C1C2=CC=CC=C2SC3=CC=CC=C13",
    "Camphorquinone": "CC1(C)C2CC1C(=O)C2=O",
    "Benzil": "O=C(C1=CC=CC=C1)C(=O)C2=CC=CC=C2",
    "Anthracene": "C1=CC=C2C=C3C=CC=CC3=CC2=C1",
    "Perylene": "C1=CC2=C3C=CC=CC3=C4C=CC=CC4=C2C=C1",
    "Anthraquinone": "O=C1C2=CC=CC=C2C(=O)C3=CC=CC=C13",
    "9,10-Phenanthrenequinone": "O=C1C2=CC=CC=C2C(=O)C3=CC=CC=C13",
    "Acridone": "O=C1C2=CC=CC=C2NC3=CC=CC=C13",
    "Michler's ketone": "CN(C)C1=CC=C(C=C1)C(=O)C2=CC=C(C=C2)N(C)C",
    "4,4'-Bis(diethylamino)benzophenone": "CCN(CC)C1=CC=C(C=C1)C(=O)C2=CC=C(C=C2)N(CC)CC",
    "Triethylamine": "CCN(CC)CC",
    "Triethanolamine": "OCCN(CCO)CCO",
    "N-Methyldiethanolamine": "CN(CCO)CCO",
    "2-Dimethylaminoethanol": "CN(C)CCO",
    "Ethyl 4-(dimethylamino)benzoate": "CCOC(=O)C1=CC=C(C=C1)N(C)C",
    "2-Ethylhexyl 4-(dimethylamino)benzoate": "CCCCC(CC)COC(=O)C1=CC=C(C=C1)N(C)C",
    "Methyl 4-(dimethylamino)benzoate": "COC(=O)C1=CC=C(C=C1)N(C)C",
    "4-(Dimethylamino)benzoic acid": "CN(C)C1=CC=C(C=C1)C(=O)O",
    "2-Mercaptoethanol": "OCCS",
    "Thioglycolic acid": "O=C(O)CS",
    "Ethyl thioglycolate": "CCOC(=O)CS",
    "1,6-Hexanedithiol": "SCCCCCCS",
    "Anthracene": "C1=CC=C2C=C3C=CC=CC3=CC2=C1",
    "Perylene": "C1=CC2=C3C=CC=CC3=C4C=CC=CC4=C2C=C1",
}


# ==================== CLIENT INSTANCE ====================
# A single PubChemClient instance for the PI family.
# Phase 5 imports this object directly and calls
# ``PI_CLIENT.fetch_single_molecule(name, role="PI_TypeI")``.

PI_CLIENT = PubChemClient(
    manual_smiles=MANUAL_SMILES,
    cache_file=CACHE_FILE,
    output_csv=OUTPUT_PI_CSV,
)


# ==================== MOLECULES LIST ====================

'''
IMPORTANT!
When adding new PIs, specify trade name(s) as `primary_names` and
insert value None as `alt_name`, or viceversa.
'''

molecules_config = [
    # ---------- TYPE I (Cleavage Photoinitiators) ----------
    {"primary_names": ["Irgacure 651", "DMPA"], "alt_name": "2,2-Dimethoxy-2-phenylacetophenone", "role": "PI_TypeI"},
    {"primary_names": ["Darocur 1173", "Omnirad 1173"], "alt_name": "2-Hydroxy-2-methylpropiophenone", "role": "PI_TypeI"},
    {"primary_names": ["Irgacure 184", "Omnirad 184"], "alt_name": "1-Hydroxycyclohexyl phenyl ketone", "role": "PI_TypeI"},
    {"primary_names": ["Irgacure 369"], "alt_name": "2-Benzyl-2-dimethylamino-1-(4-morpholinophenyl)-butan-1-one", "role": "PI_TypeI"},
    {"primary_names": ["Irgacure 907"], "alt_name": "2-Methyl-1-(4-methylthiophenyl)-2-morpholinopropan-1-one", "role": "PI_TypeI"},
    {"primary_names": ["Omnirad TPO", "Irgacure TPO", "TPO"], "alt_name": "Diphenyl(2,4,6-trimethylbenzoyl)phosphine oxide", "role": "PI_TypeI"},
    {"primary_names": ["Omnirad TPO-L", "Lucirin TPO-L", "TPO-L"], "alt_name": "Ethyl (2,4,6-trimethylbenzoyl) phenylphosphinate", "role": "PI_TypeI"},
    {"primary_names": ["Omnirad BAPO", "Irgacure 819"], "alt_name": "Bis(2,4,6-trimethylbenzoyl)-phenylphosphine oxide", "role": "PI_TypeI"},
    {"primary_names": ["Benzoin methyl ether"], "alt_name": "Benzoin methyl ether", "role": "PI_TypeI"},
    {"primary_names": ["Benzoin ethyl ether"], "alt_name": "Benzoin ethyl ether", "role": "PI_TypeI"},
    {"primary_names": ["Benzoin isopropyl ether"], "alt_name": "Benzoin isopropyl ether", "role": "PI_TypeI"},
    {"primary_names": ["Irgacure 2959"], "alt_name": "1-[4-(2-Hydroxyethoxy)-phenyl]-2-hydroxy-2-methyl-1-propanone", "role": "PI_TypeI"},
    {"primary_names": ["Irgacure 379"], "alt_name": "2-Dimethylamino-2-(4-methylbenzyl)-1-(4-morpholin-4-yl-phenyl)-butan-1-one", "role": "PI_TypeI"},
    {"primary_names": ["Omnirad Phenylglyoxylate"], "alt_name": "Phenylglyoxylic acid esters", "role": "PI_TypeI"},
    {"primary_names": ["Ivocerin"], "alt_name": "Acylgermanium compounds", "role": "PI_TypeI"},
    {"primary_names": ["Irgacure 784"], "alt_name": "Titanocenes", "role": "PI_TypeI"},

    # ---------- TYPE II (Hydrogen Abstraction Photoinitiators) ----------
    {"primary_names": ["Benzophenone", "BP"], "alt_name": "Benzophenone", "role": "PI_TypeII"},
    {"primary_names": ["Omnirad 4MBZ", "MBZ"], "alt_name": "4-Methylbenzophenone", "role": "PI_TypeII"},
    {"primary_names": ["Omnirad PBZ", "PBZ"], "alt_name": "4-Phenylbenzophenone", "role": "PI_TypeII"},
    {"primary_names": ["Omnirad OMBB", "OMBB"], "alt_name": "Methyl o-benzoylbenzoate", "role": "PI_TypeII"},
    {"primary_names": ["Omnirad ITX", "ITX"], "alt_name": "2-Isopropylthioxanthone", "role": "PI_TypeII"},
    {"primary_names": ["TX"], "alt_name": "Thioxanthone", "role": "PI_TypeII"},
    {"primary_names": ["DETX"], "alt_name": "2,4-Diethylthioxanthone", "role": "PI_TypeII"},
    {"primary_names": ["Camphorquinone", "CQ"], "alt_name": "Camphorquinone", "role": "PI_TypeII"},
    {"primary_names": ["Benzil", "Dibenzoyl"], "alt_name": "Benzil", "role": "PI_TypeII"},
    {"primary_names": ["Anthraquinone"], "alt_name": "Anthraquinone", "role": "PI_TypeII"},
    {"primary_names": ["9,10-Phenanthrenequinone"], "alt_name": "9,10-Phenanthrenequinone", "role": "PI_TypeII"},
    {"primary_names": ["Acridone"], "alt_name": "Acridone", "role": "PI_TypeII"},
    {"primary_names": ["Michler's ketone"], "alt_name": "4,4'-Bis(dimethylamino)benzophenone", "role": "PI_TypeII"},
    {"primary_names": ["4,4'-Bis(diethylamino)benzophenone"], "alt_name": "4,4'-Bis(diethylamino)benzophenone", "role": "PI_TypeII"},
    {"primary_names": ["3,3'-carbonylbiscoumarin"], "alt_name": "Ketocoumarins", "role": "PI_TypeII"},
    {"primary_names": ["Eosin Y"], "alt_name": "Eosin Y", "role": "PI_TypeII"},
    {"primary_names": ["Methylene blue"], "alt_name": "Methylene blue", "role": "PI_TypeII"},
    {"primary_names": ["Rose bengal"], "alt_name": "Rose bengal", "role": "PI_TypeII"},

    # ---------- CO-INITIATORS / SYNERGISTS (Tertiary Amines) ----------
    {"primary_names": ["TEA"], "alt_name": "Triethylamine", "role": "co-initiator"},
    {"primary_names": ["TEOA"], "alt_name": "Triethanolamine", "role": "co-initiator"},
    {"primary_names": ["MDEA"], "alt_name": "N-Methyldiethanolamine", "role": "co-initiator"},
    {"primary_names": ["DMAE"], "alt_name": "2-Dimethylaminoethanol", "role": "co-initiator"},
    {"primary_names": ["DMBA"], "alt_name": "N,N-Dimethylbenzylamine", "role": "co-initiator"},
    {"primary_names": ["PDEA"], "alt_name": "N-Phenyldiethanolamine", "role": "co-initiator"},
    {"primary_names": ["DIPEA"], "alt_name": "Diisopropylethylamine", "role": "co-initiator"},
    {"primary_names": ["DABCO"], "alt_name": "1,4-Diazabicyclo[2.2.2]octane", "role": "co-initiator"},

    # ---------- CO-INITIATORS (Aromatic Amine Esters) ----------
    {"primary_names": ["EDB", "Ethyl-4-Dimethylaminobenzoate"], "alt_name": "Ethyl 4-(dimethylamino)benzoate", "role": "co-initiator"},
    {"primary_names": ["EHA", "Octyl-4-Dimethylaminobenzoate"], "alt_name": "2-Ethylhexyl 4-(dimethylamino)benzoate", "role": "co-initiator"},
    {"primary_names": ["MDB"], "alt_name": "Methyl 4-(dimethylamino)benzoate", "role": "co-initiator"},
    {"primary_names": ["BDB"], "alt_name": "Butyl 4-(dimethylamino)benzoate", "role": "co-initiator"},
    {"primary_names": ["DMABA"], "alt_name": "4-(Dimethylamino)benzoic acid", "role": "co-initiator"},
    {"primary_names": ["DMAB"], "alt_name": "Dimethylaminobenzaldehyde", "role": "co-initiator"},

    # ---------- CO-INITIATORS (Thiols / Mercaptans) ----------
    {"primary_names": ["2-Mercaptoethanol", "Thioglycol", "ME"], "alt_name": "Thioglycol", "role": "co-initiator"},
    {"primary_names": ["Thioglycolic acid", "Mercaptoacetic acid"], "alt_name": "Mercaptoacetic acid", "role": "co-initiator"},
    {"primary_names": ["Ethyl thioglycolate"], "alt_name": None, "role": "co-initiator"},
    {"primary_names": ["PETMP"], "alt_name": "Pentaerythritol tetrakis(3-mercaptopropionate)", "role": "co-initiator"},
    {"primary_names": ["TMPMP"], "alt_name": "Trimethylolpropane tris(3-mercaptopropionate)", "role": "co-initiator"},
    {"primary_names": ["HDT", "1,6-Hexanedithiol"], "alt_name": "1,6-Hexanedithiol", "role": "co-initiator"},

    # ---------- OTHER SYNERGISTS / SENSITIZERS ----------
    {"primary_names": ["Anthracene"], "alt_name": "Anthracene", "role": "co-initiator"},
    {"primary_names": ["Perylene"], "alt_name": "Perylene", "role": "co-initiator"},
]


# ==================== BACKWARD-COMPAT WRAPPER ====================
def fetch_single_PI(name: str) -> str | None:
    '''
    Legacy thin wrapper. 
    Returns only the SMILES string (or ``None``).

    Kept for backward compatibility with any Phase 1 code that still
    imports it. 
    New code (Phase 5) should call
    ``PI_CLIENT.fetch_single_molecule(name, role="PI_TypeI")`` directly
    to get the full ``{"name", "smiles", "role"}`` dict.
    '''
    result = PI_CLIENT.fetch_single_molecule(name, role="PI_TypeI")
    return result["smiles"] if result else None


# ==================== MAIN EXECUTION ====================
def main():
    '''
    Batch entry point: resolve the canonical SMILES for every molecule
    declared in ``molecules_config`` and persist the results to
    ``data/molecules_PIs.csv``.

    Workflow
    --------
    1. Iterate over ``molecules_config`` (list of dicts, each with
       ``primary_names``, optional ``alt_name`` and ``role``).
    2. For each entry, call :meth:`PubChemClient.get_smiles_robust`,
       which resolves the SMILES using, in order of preference:
         (a) the local cache ``data/smiles_cache_PIs.csv``,
         (b) the manual fallback dictionary ``MANUAL_SMILES``,
         (c) the PubChem PUG REST API (name -> CID -> CanonicalSMILES).
    3. Collect one record per molecule: ``{"name", "smiles", "role"}``.
    4. Overwrite ``data/molecules_PIs.csv`` (fully rewritten every run,
       so it always reflects the current configuration).
    5. Print a summary (found / total, list of missing molecules) and a
       preview of the first 10 rows.

    Reads
    -----
    - Module constant ``molecules_config``.
    - ``data/smiles_cache_PIs.csv`` (via ``PI_CLIENT``).

    Writes
    ------
    - ``data/molecules_PIs.csv``     — final dataset consumed by Phases 2–5.
    - ``data/smiles_cache_PIs.csv``  — updated on every successful lookup.

    Notes
    -----
    Must be run from the project root:
        ``python -m phase1.fetch_molecules_PIs``
    Re-run every time ``molecules_config`` changes.
    '''
    print('Starting fetching SMILES using PubChem REST API + manual fallback...')
    print(f'Total molecules to process: {len(molecules_config)}\n')

    results = []
    for idx, item in enumerate(molecules_config, 1):
        primary_names = item["primary_names"]
        alt = item["alt_name"]
        role = item["role"]

        print(f'[{idx}/{len(molecules_config)}] Processing: {", ".join(primary_names)} (role: {role})')

        smiles, used_name = PI_CLIENT.get_smiles_robust(primary_names, alt)
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
    df_final.to_csv(OUTPUT_PI_CSV, index=False)

    found = df_final['smiles'].notna().sum()
    total = len(df_final)
    print(f'\n[COMPLETED] CSV saved as "{OUTPUT_PI_CSV}"')
    print(f'    Found molecules: {found} over {total}')
    if found < total:
        missing = df_final[df_final['smiles'].isna()]['name'].tolist()
        print(f'    Not found: {missing}')

    print('\nPreview of generated CSV:')
    print(df_final.head(10).to_string())


if __name__ == "__main__":
    main()

'''
========================================================================
The REST + manual fallback approach was adopted because:
========================================================================

    - The REST API allows us to explicitly request the CanonicalSMILES, which is always available.
    - The manual fallback (a dictionary of pre-existing SMILES) was added 
      as a safety net for the most common molecules, in case the REST API 
      failed (due to unrecognized names or network issues).


🎯 Summary
===========

Question	                            Answer
--------                                ------
What is the difference?	                pubchempy is a convenient but limited wrapper; 
                                        the REST API is more flexible and reliable for specific use cases.
Why didn't we use REST right away?	    Because pubchempy is the standard choice and usually works; 
                                        the issue only came to light after testing.
What did we learn?	                    To recognize when to abandon a library and use the API directly.
'''

