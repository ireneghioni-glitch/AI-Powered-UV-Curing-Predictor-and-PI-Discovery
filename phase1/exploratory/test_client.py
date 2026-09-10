"""Throwaway sanity-check for shared/pubchem_client.py. Delete after use."""
import shutil
import tempfile
from pathlib import Path

from shared.pubchem_client import PubChemClient, _load_csv_safe, _append_molecule


# ---- helper CSV (cross-platform) ----
tmp = Path(tempfile.mkdtemp(prefix="pubchem_test_"))
print(f"Working dir: {tmp}")

csv_out = tmp / "molecules_test.csv"
csv_cache = tmp / "cache_test.csv"

# 1. _append_molecule + dedup case-insensitive
_append_molecule(csv_out, "Aspirin", "CC(=O)Oc1ccccc1C(=O)O", "test")
_append_molecule(csv_out, "aspirin", "CC(=O)Oc1ccccc1C(=O)O", "test")
df = _load_csv_safe(csv_out)
assert len(df) == 1, f"dedup failed: {len(df)} rows"
print("[OK] 1. _append_molecule dedup")

# 2. _load_csv_safe su file inesistente
empty = _load_csv_safe(tmp / "does_not_exist.csv")
assert list(empty.columns) == ["name", "smiles", "role"] and len(empty) == 0
print("[OK] 2. _load_csv_safe empty")

# 3. Client construction
client = PubChemClient(
    manual_smiles={
        "Benzophenone": "O=C(C1=CC=CC=C1)C2=CC=CC=C2",
        "TMPTA": "C=C(C)C(=O)OCC(COC(=O)C(=C)C)(COC(=O)C(=C)C)COC(=O)C(=C)C",
    },
    cache_file=csv_cache,
    output_csv=csv_out,
)
assert client.manual_smiles_lower["benzophenone"][0] == "Benzophenone"
print("[OK] 3. Client construction + case-insensitive index")

# 4. fetch_single_molecule — manual dict path (no network)
r = client.fetch_single_molecule("benzophenone", role="PI_TypeII")
assert r == {"name": "Benzophenone", "smiles": "O=C(C1=CC=CC=C1)C2=CC=CC=C2", "role": "PI_TypeII"}, r
print("[OK] 4. fetch_single_molecule via manual dict")

# 5. Idempotency — secondo call non deve aggiungere righe
n_before = len(_load_csv_safe(csv_out))
client.fetch_single_molecule("BENZOPHENONE", role="PI_TypeII")
n_after = len(_load_csv_safe(csv_out))
assert n_before == n_after, f"duplicate appended: {n_before} -> {n_after}"
print("[OK] 5. Idempotency (no duplicate rows)")


# 6. fetch_single_molecule — API path (network), CSV pulito
csv_api = tmp / "molecules_api.csv"
client_api = PubChemClient(
    manual_smiles={},
    cache_file=csv_cache,
    output_csv=csv_api,
)
r2 = client_api.fetch_single_molecule("aspirin", role="monomer")
assert r2 is not None and r2["role"] == "monomer", r2
print(f"[OK] 6. fetch_single_molecule via PubChem API -> {r2['smiles']}")

# 6b. Role persistence — il role esistente nel CSV vince sul parametro
r2b = client_api.fetch_single_molecule("aspirin", role="co-initiator")
assert r2b["role"] == "monomer", f"role was overwritten: {r2b}"
print("[OK] 6b. Existing role in CSV takes precedence (persistence)")

# 7. Molecola inesistente → None, non crash
r3 = client.fetch_single_molecule("xyz_definitely_not_a_molecule_42")
assert r3 is None
print("[OK] 7. Unknown molecule returns None safely")

# 8. get_smiles_robust — manual path
smiles, name = client.get_smiles_robust(["Benzophenone"])
assert smiles.startswith("O=C") and name == "Benzophenone"
print("[OK] 8. get_smiles_robust via manual dict")

# 9. get_smiles_robust — fallback su alt_name
smiles, name = client.get_smiles_robust(["fake_trade_name_xyz"], alt_name="Benzophenone")
assert smiles.startswith("O=C"), (smiles, name)
print("[OK] 9. get_smiles_robust fallback to alt_name")

print("\n[SUCCESS] All checkpoints passed.")

# Deletes the temporary directory even if a test fails.
shutil.rmtree(tmp, ignore_errors=True)