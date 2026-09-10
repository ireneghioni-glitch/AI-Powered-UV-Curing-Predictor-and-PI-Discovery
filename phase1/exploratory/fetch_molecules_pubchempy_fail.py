import pubchempy as pcp
import pandas as pd
import time
import os
from pathlib import Path


# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CACHE_PI_FILE = DATA_DIR / "smiles_cache.csv"
OUTPUT_PI_CSV = DATA_DIR / "molecules_PIs.csv"


# ==================== CACHE FUNCTIONS ====================

def load_cache():
    if os.path.exists(CACHE_PI_FILE):
        df = pd.read_csv(CACHE_PI_FILE)
        for col in ["iupac_name", "trade_name", "smiles"]:
            if col not in df.columns:
                df[col] = None
        return df[["iupac_name", "trade_name", "smiles"]]
    return pd.DataFrame(columns=["iupac_name", "trade_name", "smiles"])

def save_cache(df):
    df[["iupac_name", "trade_name", "smiles"]].to_csv(CACHE_PI_FILE, index=False)

def get_smiles_robust(primary_names, alt_name=None):
    df = load_cache()

    # 1. Check cache
    all_names = list(primary_names)
    if alt_name:
        all_names.append(alt_name)
    for name in all_names:
        if name is None:
            continue
        mask = (df["trade_name"] == name) | (df["iupac_name"] == name)
        if mask.any():
            print(f'{name} found in cache.')
            row = df[mask].iloc[0]
            smiles = row["smiles"]
            # Se in cache c'è NaN, restituiamo None
            if pd.isna(smiles):
                return None, name
            return smiles, name

    # 2. Try each primary name
    for primary in primary_names:
        print(f'Searching for {primary} on PubChem...')
        try:
            time.sleep(0.5)
            compounds = pcp.get_compounds(primary, "name")
            if compounds:
                # Prendiamo il primo composto
                cpd = compounds[0]
                # Tentiamo isomeric_smiles, poi canonical_smiles
                smiles = cpd.isomeric_smiles
                if smiles is None:
                    smiles = cpd.canonical_smiles  # fallback
                if smiles is None:
                    print(f'    [WARN] No SMILES for {primary} (CID: {cpd.cid}), skipping.')
                    continue
                # Salva in cache
                new_row = pd.DataFrame([[None, primary, smiles]],
                                       columns=["iupac_name", "trade_name", "smiles"])
                df = pd.concat([df, new_row], ignore_index=True)
                save_cache(df)
                print(f'    [OK] Found with {primary} (CID: {cpd.cid})')
                return smiles, primary
        except Exception as e:
            print(f'    [FAIL] Error fetching {primary}: {e}')

    # 3. Fallback alt_name (IUPAC)
    if alt_name:
        print(f'    [INFO] Fallback on {alt_name} (IUPAC)...')
        try:
            time.sleep(0.5)
            compounds = pcp.get_compounds(alt_name, "name")
            if compounds:
                cpd = compounds[0]
                smiles = cpd.isomeric_smiles
                if smiles is None:
                    smiles = cpd.canonical_smiles
                if smiles is None:
                    print(f'    [WARN] No SMILES for {alt_name} (CID: {cpd.cid}), skipping.')
                else:
                    first_primary = primary_names[0] if primary_names else None
                    new_row = pd.DataFrame([[alt_name, first_primary, smiles]],
                                           columns=["iupac_name", "trade_name", "smiles"])
                    df = pd.concat([df, new_row], ignore_index=True)
                    save_cache(df)
                    print(f'    [OK] Found with {alt_name} (CID: {cpd.cid})')
                    return smiles, alt_name
        except Exception as e:
            print(f'    [FAIL] Error in fallback fetching {alt_name}: {e}')

    # 4. Se ancora nulla, prova a cercare per CID per le molecole note
    # (questo è un fallback d'emergenza per i casi più comuni)
    known_cids = {
        "Benzophenone": 3102,
        "Irgacure 651": 638686,  # 2,2-Dimethoxy-2-phenylacetophenone
        "Irgacure 184": 12941,   # 1-Hydroxycyclohexyl phenyl ketone
        "Camphorquinone": 160522,
        "Benzil": 8651,
        "Anthracene": 8418,
        "Perylene": 6984,
    }
    for name in primary_names:
        if name in known_cids:
            cid = known_cids[name]
            print(f'    [INFO] Trying fallback by CID {cid} for {name}...')
            try:
                time.sleep(0.5)
                cpd = pcp.Compound.from_cid(cid)
                if cpd:
                    smiles = cpd.isomeric_smiles or cpd.canonical_smiles
                    if smiles:
                        first_primary = primary_names[0] if primary_names else None
                        new_row = pd.DataFrame([[alt_name, first_primary, smiles]],
                                               columns=["iupac_name", "trade_name", "smiles"])
                        df = pd.concat([df, new_row], ignore_index=True)
                        save_cache(df)
                        print(f'    [OK] Found by CID {cid}')
                        return smiles, name
            except Exception as e:
                print(f'    [FAIL] CID fallback failed: {e}')

    print(f'    No SMILES found for {", ".join(primary_names)} nor {alt_name}; molecule skipped.')
    return None, None


# ==================== MOLECULES LIST (your config unchanged) ====================

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


# ==================== MAIN EXECUTION ====================

print('Starting fetching SMILES with automatic fallback...')
print(f'Total molecules to process: {len(molecules_config)}\n')

results = []
for idx, item in enumerate(molecules_config, 1):
    primary_names = item["primary_names"]
    alt = item["alt_name"]
    role = item["role"]

    print(f'[{idx}/{len(molecules_config)}] Processing: {", ".join(primary_names)} (role: {role})')

    smiles, used_name = get_smiles_robust(primary_names, alt)

    if smiles is not None:
        print(f'    DEBUG: smiles = {smiles[:80]}... (type: {type(smiles).__name__}) used_name = {used_name}')
    else:
        print(f'    DEBUG: smiles = None, used_name = {used_name}')

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


'''
========================================================================
WHY IT DOESN'T WORK
========================================================================

The problem was that for certain names (e.g., "Irgacure 651"), PubChem returned 
a compound that lacked the `isomeric_smiles` field (it was `None`). 

This is because PubChem stores multiple records for the same substance 
(e.g., a mixture, a salt, or an isomer). A search by name often returns the 
first record, which is not always the one containing the SMILES string.

Furthermore, the returned CID (e.g., 90571 for Irgacure 651) was not the CID 
for the active ingredient (which is 638686). Thus, the compound found was 
technically a valid record, but it lacked SMILES data.
'''