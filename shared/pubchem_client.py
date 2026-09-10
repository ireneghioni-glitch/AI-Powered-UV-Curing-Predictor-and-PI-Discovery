'''
Shared PubChem REST client + manual fallback + safe CSV persistence.

Used by phase1/fetch_molecules_PIs.py, phase1/fetch_molecules_monomers.py
and phase5 PredictorState (runtime single-molecule lookup).
'''

from __future__ import annotations
import threading
from pathlib import Path
import pandas as pd
import requests

# Single lock serialising ALL CSV writes across all client instances.
_CSV_LOCK = threading.Lock()

# Single source of truth for the molecules CSV schema.
_CSV_COLUMNS = ["name", "smiles", "role"]


# ==================== THREADING ====================
def _load_csv_safe(path: Path) -> pd.DataFrame:
    '''Load a molecules CSV, returning an empty DataFrame 
    on missing/empty/corrupt file.'''
    if not path.exists():
        return pd.DataFrame(columns=_CSV_COLUMNS)
    try:
        df = pd.read_csv(path)
        return df if not df.empty else pd.DataFrame(columns=_CSV_COLUMNS)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=_CSV_COLUMNS)


# prevents the final tool from crashing when updating the CSV
def _append_molecule(path: Path, name: str, smiles: str, role: str) -> None:
    '''Thread-safe append with case-insensitive dedup.'''
    with _CSV_LOCK:
        df = _load_csv_safe(path)
        if not df.empty and (df["name"].astype(str).str.lower() == name.lower()).any():
            return  # already there
        df = pd.concat(
            [df, pd.DataFrame([[name, smiles, role]], columns=_CSV_COLUMNS)],
            ignore_index=True,
        )
        df.to_csv(path, index=False)


# ==================== PUBCHEM CLIENT CLASS ====================
class PubChemClient:
    '''
    Resolve molecule names to canonical SMILES with a 3-tier fallback:
      1. cache CSV        (local, no network)
      2. manual dictionary (offline, curated)
      3. PubChem REST API (name -> CID -> CanonicalSMILES)
    '''
    _PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"

    def __init__(
            self,
            manual_smiles: dict[str, str],
            cache_file: Path,
            output_csv: Path,
            request_timeout: float = 10.0
    ) -> None:
        self.manual_smiles = dict(manual_smiles)
        self.manual_smiles_lower = {
            k.lower(): (k, v) for k, v in manual_smiles.items()
        }
        self.cache_file = Path(cache_file)
        self.output_csv = Path(output_csv)
        self.request_timeout = request_timeout

    def _get_cid_from_name(self, name: str) -> int | None:
        url = f"{self._PUBCHEM_BASE}/name/{name}/cids/TXT"
        try:
            r = requests.get(url, timeout=self.request_timeout)
            if r.status_code == 200:
                cids = r.text.strip().split()
                if cids:
                    return int(cids[0])
        except Exception as exc:
            print(f"    [API ERROR] {exc}")
        return None

    def _get_smiles_from_cid(self, cid: int) -> str | None:
        url = f"{self._PUBCHEM_BASE}/cid/{cid}/property/CanonicalSMILES/TXT"
        try:
            r = requests.get(url, timeout=self.request_timeout)
            if r.status_code == 200:
                smiles = r.text.strip()
                return smiles or None
        except Exception as exc:
            print(f"    [API ERROR] {exc}")
        return None

    def _load_cache(self) -> pd.DataFrame:
        if self.cache_file.exists():
            try:
                return pd.read_csv(self.cache_file)
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                pass
        return pd.DataFrame(columns=["name", "smiles"])

    def _save_cache(self, df: pd.DataFrame) -> None:
        df.to_csv(self.cache_file, index=False)

    def _save_to_cache(self, cache_df: pd.DataFrame, name: str, smiles: str) -> None:
        if name and smiles:
            new_row = pd.DataFrame([[name, smiles]], columns=["name", "smiles"])
            self._save_cache(pd.concat([cache_df, new_row], ignore_index=True))
    
    # Why is the cache separate from the output CSV? 
    # Because they serve two different purposes:

    # - cache (smiles_cache_PIs.csv): stores every successful lookup, 
    #   including those for molecules that do not end up in the final 
    #   dataset. It is a network optimization.
    # - output (molecules_PIs.csv): is the cleaned dataset, the curated 
    #   list of molecules of interest.

    # They coexist within the client because they serve the same purpose 
    # (resolving SMILES), but they remain distinct files.
    
    # copy from former `fetch_molecules_*.py` with variables names
    # changed according to PubChemClient class
    def get_smiles_robust(
        self,
        primary_names: list[str],
        alt_name: str | None = None,
    ) -> tuple[str | None, str | None]:
        '''
        Resolve SMILES for a molecule described by one or more trade names
        and an optional IUPAC fallback.

        Resolution order
        ----------------
        1. Cache CSV (no network).
        2. Manual dictionary (no network).
        3. PubChem REST: ``name -> CID -> CanonicalSMILES``.

        Parameters
        ----------
        primary_names : list[str]
            Trade names to try in order (e.g. ``["Omnirad TPO", "Irgacure TPO", "TPO"]``).
        alt_name : str | None
            IUPAC fallback, tried only if every ``primary_name`` fails.

        Returns
        -------
        (smiles, used_name) : tuple[str | None, str | None]
            ``(None, None)`` if the molecule could not be resolved.
        '''
        cache_df = self._load_cache()
        manual = self.manual_smiles

        # 1. cache lookup
        for name in primary_names + ([alt_name] if alt_name else []):
            if name and name in cache_df["name"].values:
                print(f"    [CACHE] Found {name} in cache")
                smiles = cache_df.loc[cache_df["name"] == name, "smiles"].iloc[0]
                return smiles, name

        # 2. primary names (manual + API)
        for primary in primary_names:
            print(f"Searching for {primary}...")
            if primary in manual:
                smiles = manual[primary]
                print("    [OK] Found in manual fallback")
                self._save_to_cache(cache_df, primary, smiles)
                return smiles, primary

            cid = self._get_cid_from_name(primary)
            if cid:
                smiles = self._get_smiles_from_cid(cid)
                if smiles:
                    print(f"    [OK] Found via CID {cid}")
                    self._save_to_cache(cache_df, primary, smiles)
                    return smiles, primary
                print(f"    [WARN] CID {cid} has no SMILES")
            else:
                print(f"    [WARN] No CID found for {primary}")

        # 3. alt_name (IUPAC)
        if alt_name:
            print(f"    [INFO] Fallback on {alt_name}...")
            if alt_name in manual:
                smiles = manual[alt_name]
                print("    [OK] Found in manual fallback")
                self._save_to_cache(cache_df, alt_name, smiles)
                return smiles, alt_name

            cid = self._get_cid_from_name(alt_name)
            if cid:
                smiles = self._get_smiles_from_cid(cid)
                if smiles:
                    print(f"    [OK] Found via CID {cid}")
                    self._save_to_cache(cache_df, alt_name, smiles)
                    return smiles, alt_name
                print(f"    [WARN] CID {cid} has no SMILES")
            else:
                print(f"    [WARN] No CID found for {alt_name}")

        print(f'    No SMILES found for {", ".join(primary_names)} nor {alt_name}; molecule skipped.')
        return None, None

    def fetch_single_molecule(
        self,
        name: str,
        role: str = "unknown",
        csv_path: Path | None = None,
    ) -> dict | None:
        '''
        Runtime single-molecule lookup, used by Phase 5 (Reflex web app).

        Compared to the former ``fetch_single_PI``:
          1. ``role`` and ``csv_path`` are parameters (no hardcoding).
          2. Uses the safe helpers ``_load_csv_safe`` / ``_append_molecule``
             instead of raw ``pd.read_csv`` / ``to_csv``.
          3. Returns a full ``{"name", "smiles", "role"}`` dict, not a bare
             SMILES string.

        On success the molecule is appended to ``csv_path`` (default
        ``self.output_csv``), deduplicated case-insensitively, so future
        lookups skip the network.

        Parameters
        ----------
        name : str
            User-typed name (trade name, IUPAC, or synonym).
        role : str, default "unknown"
            Chemical role assigned if the molecule is new. Examples:
            ``"PI_TypeI"``, ``"PI_TypeII"``, ``"co-initiator"``, ``"monomer"``.
        csv_path : Path | None
            Override the destination CSV (useful for monomer vs PI families).

        Returns
        -------
        dict | None
            ``{"name": str, "smiles": str, "role": str}`` on success,
            ``None`` if the molecule could not be resolved.
        '''
        if not name or not name.strip():
            return None
        key = name.strip()
        key_lower = key.lower()
        target_csv = Path(csv_path) if csv_path else self.output_csv

        # 1. already known
        df = _load_csv_safe(target_csv)
        if not df.empty:
            match = df[df["name"].astype(str).str.lower() == key_lower]
            if not match.empty:
                row = match.iloc[0]
                return {
                    "name": row["name"],
                    "smiles": row["smiles"],
                    "role": row.get("role", role),
                }

        # 2. manual dict (no network)
        if key_lower in self.manual_smiles_lower:
            canonical, smiles = self.manual_smiles_lower[key_lower]
            _append_molecule(target_csv, canonical, smiles, role)
            return {"name": canonical, "smiles": smiles, "role": role}

        # 3. PubChem REST
        try:
            smiles, used_name = self.get_smiles_robust([key], None)
        except Exception as exc:
            print(f"[fetch_single_molecule] API error for '{key}': {exc}")
            return None

        if not smiles:
            return None

        resolved = used_name or key
        _append_molecule(target_csv, resolved, smiles, role)
        return {"name": resolved, "smiles": smiles, "role": role}        