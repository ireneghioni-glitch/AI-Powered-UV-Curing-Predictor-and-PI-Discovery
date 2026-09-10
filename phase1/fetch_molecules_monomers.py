'''
IMPORTANT!
This script has to be run again AFTER EACH TIME THE LIST `molecules_config_monomers`
IS UPDATED WITH NEW MOLECULES.'''

import time
import pandas as pd
import requests
from pathlib import Path


# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MONO_CSV = DATA_DIR / "molecules_monomers.csv"
CACHE_FILE = DATA_DIR / "smiles_cache_monomers.csv"

# ==================== CACHE FUNCTIONS ====================

def load_cache():
    if CACHE_FILE.exists():
        return pd.read_csv(CACHE_FILE)
    return pd.DataFrame(columns=["name", "smiles"])

def save_cache(df):
    df.to_csv(CACHE_FILE, index=False)

def _save_to_cache(cache_df, name, smiles):
    """Helper function to save a found SMILES to cache."""
    if name and smiles:
        new_row = pd.DataFrame([[name, smiles]], columns=["name", "smiles"])
        updated_df = pd.concat([cache_df, new_row], ignore_index=True)
        save_cache(updated_df)


# ==================== FUNCTIONS ====================

def get_smiles_from_cid(cid):
    """Retrieves the Canonical SMILES for a CID using the PubChem REST API."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/TXT"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            smiles = response.text.strip()
            return smiles if smiles else None
    except Exception as e:
        print(f'    [API ERROR] {e}')
    return None

def get_cid_from_name(name):
    """Retrieves the CID of a compound given its name, using the REST API."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/TXT"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            cids = response.text.strip().split()
            if cids:
                return int(cids[0])
    except Exception as e:
        print(f'    [API ERROR] {e}')
    return None

def get_smiles_robust(primary_names, alt_name=None):
    """
    Search for the SMILES:
    1. Try each primary_name to obtain the CID, then the SMILES.
    2. If that fails, try the alt_name.
    3. Manual fallback for known molecules (known CID + SMILES).
    """
    # load cache
    cache_df = load_cache()

    # check if name is already in cache
    for name in primary_names + ([alt_name] if alt_name else []):
        if name and name in cache_df['name'].values:
            print(f'    [CACHE] Found {name} in cache')
            return cache_df[cache_df['name'] == name]['smiles'].iloc[0], name
    
    # Manual fallback for the most common molecules
    manual_smiles_monomers = {
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

    # Test each primary name (trade name)
    for primary in primary_names:
        print(f'Searching for {primary}...')
        # Manual Fallback
        if primary in manual_smiles_monomers:
            smiles = manual_smiles_monomers[primary]
            used_name = primary
            print(f'    [OK] Found in manual fallback')
            # SAVE TO CACHE
            _save_to_cache(cache_df, used_name, smiles)
            return smiles, used_name

        # Get CID from name (API)
        cid = get_cid_from_name(primary)
        if cid:
            smiles = get_smiles_from_cid(cid)
            if smiles:
                used_name = primary
                print(f'    [OK] Found via CID {cid}')
                # SAVE TO CACHE
                _save_to_cache(cache_df, used_name, smiles)
                return smiles, used_name
            else:
                print(f'    [WARN] CID {cid} has no SMILES')
        else:
            print(f'    [WARN] No CID found for {primary}')

    # Fallback on alt_name (IUPAC)
    if alt_name:
        print(f'    [INFO] Fallback on {alt_name}...')
        if alt_name in manual_smiles_monomers:
            smiles = manual_smiles_monomers[alt_name]
            used_name = alt_name
            print(f'    [OK] Found in manual fallback')
            _save_to_cache(cache_df, used_name, smiles)
            return smiles, used_name

        cid = get_cid_from_name(alt_name)
        if cid:
            smiles = get_smiles_from_cid(cid)
            if smiles:
                used_name = alt_name
                print(f'    [OK] Found via CID {cid}')
                _save_to_cache(cache_df, used_name, smiles)
                return smiles, used_name
            else:
                print(f'    [WARN] CID {cid} has no SMILES')
        else:
            print(f'    [WARN] No CID found for {alt_name}')

    # MONOMER NOT FOUND
    print(f'    No SMILES found for {", ".join(primary_names)} nor {alt_name}; molecule skipped.')
    return None, None


# ==================== MOLECULES LIST ====================

'''
IMPORTANT!
When adding new monomers, specify trade name(s) as `primary_names` and
insert value None as `alt_name`,
or viceversa.'''

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


# ==================== MAIN EXECUTION ====================

print('Starting fetching SMILES using PubChem REST API + manual fallback...')
print(f'Total molecules to process: {len(molecules_config_monomers)}\n')

results = []
for idx, item in enumerate(molecules_config_monomers, 1):
    primary_names = item["primary_names"]
    alt = item["alt_name"]
    role = item["role"]

    print(f'[{idx}/{len(molecules_config_monomers)}] Processing: {", ".join(primary_names)} (role: {role})')

    smiles, used_name = get_smiles_robust(primary_names, alt)
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