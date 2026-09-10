# Future Refactor: Externalizing the Molecule Configuration

#### Document Purpose
This document describes a planned refactor for Phase 1 (and its downstream consumers in Phase 5) [747]. The goal is to move the hardcoded `molecules_config` list out of the Python source code and into a versioned, external configuration file [747]. This will allow new molecules to be added **without modifying any Python code**, which is essential for:
* A scalable MVP that grows over time [747].
* The Phase 5 web app to dynamically add user-requested molecules [747].
* Cleaner separation between *data* and *logic* [747].
* Easier version control and collaboration [747].

This document is a **plan**, not an implementation [747]. It should be read before starting the refactor [747].

---

#### 1. The Problem with the Current Design
In both `fetch_molecules_PIs.py` and `fetch_molecules_monomers.py`, the list of molecules is defined **inline** in the Python source [748]:

```python
molecules_config = [
    {"primary_names": ["Irgacure 651", "DMPA"], "alt_name": "...", "role": "PI_TypeI"},
    ...
]
```

##### Why this is a limitation
| Issue | Impact |
| :--- | :--- |
| **Code change required** | Adding a new molecule means editing a `.py` file [749]. |
| **Not user-friendly** | Non-developers cannot add molecules safely [749]. |
| **No runtime mutation** | A web app (Phase 5) cannot append to a hardcoded Python list on disk [749]. |
| **Merging conflicts** | Two developers adding molecules on different branches will hit Git conflicts [749]. |
| **Tight coupling** | Logic (fetching SMILES) and data (list of molecules) are mixed [749]. |

The best practice is to externalize the configuration into a separate file [749]. The Python code reads the file at startup; new molecules are appended to the file, not to the code [749].

---

#### 2. Choosing the Right Format
Three common formats exist for external configuration [749]:

| Format | Pros | Cons | Best for |
| :--- | :--- | :--- | :--- |
| **CSV** | Human-editable in Excel, Git-friendly, simple parsing | Cannot represent nested lists cleanly | Flat, tabular data (e.g., a single name per row) [749] |
| **JSON** | Supports nested structures (lists, dicts), universal | Slightly verbose to edit by hand | Our case: `primary_names` is a list [749] |
| **YAML** | Very readable, supports comments | Requires an external library (PyYAML) | When human readability is critical [749] |

##### Our choice: JSON
We choose JSON because [749]:
* Our molecule entries contain a list (`primary_names`) and an optional string (`alt_name`) [749].
* JSON is natively supported by Python's `json` module — no extra dependency [749].
* JSON files can be validated by schema if needed [749].

---

#### 3. The Target Architecture

##### 3.1 File layout
```text
phase1/
├── data/
│   ├── molecules_config.json          # NEW: configuration file [749]
│   ├── molecules_PIs.csv              # existing: resolved SMILES for PIs [749]
│   ├── molecules_monomers.csv         # existing: resolved SMILES for monomers [749]
│   ├── smiles_cache_PIs.csv           # existing cache [749]
│   └── smiles_cache_monomers.csv      # existing cache [749]
├── fetch_molecules_PIs.py             # modified: reads JSON [749]
└── fetch_molecules_monomers.py        # modified: reads JSON [749]
```

##### 3.2 JSON schema
```json
{
  "PIs": [
    {
      "primary_names": ["Irgacure 651", "DMPA"],
      "alt_name": "2,2-Dimethoxy-2-phenylacetophenone",
      "role": "PI_TypeI"
    }
  ],
  "monomers": [
    {
      "primary_names": ["TMPTA"],
      "alt_name": "Trimethylolpropane triacrylate",
      "role": "monomer"
    }
  ]
}
```

##### 3.3 Data flow
```text
┌───────────────────────────────────────────────────────────────────┐
│  molecules_config.json                                             │
│  (source of truth for the list of molecules to fetch)              │
└───────────────────────────────────────────────────────────────────┘
                          │
                          │ read at startup
                          ▼
┌───────────────────────────────────────────────────────────────────┐
│  fetch_molecules_PIs.py / fetch_molecules_monomers.py              │
│  - iterate over the config                                          │
│  - call get_smiles_robust() for each molecule                      │
│  - write results to molecules_*.csv                                │
└───────────────────────────────────────────────────────────────────┘
                          │
                          │ produces
                          ▼
┌───────────────────────────────────────────────────────────────────┐
│  molecules_PIs.csv / molecules_monomers.csv                        │
│  (resolved SMILES for downstream phases)                           │
└───────────────────────────────────────────────────────────────────┘
                          │
                          │ consumed by
                          ▼
┌───────────────────────────────────────────────────────────────────┐
│  Phase 2 / 3 / 4 / 5                                               │
└───────────────────────────────────────────────────────────────────┘
```

---

#### 4. Step‑by‑Step Refactor Plan

##### Step 4.1 — Extract the current list to JSON
1. Open `fetch_molecules_PIs.py`. Copy the `molecules_config` list [749].
2. Paste it into a new file `phase1/data/molecules_config.json` under the key `"PIs"` [749].
3. Do the same for `fetch_molecules_monomers.py` under the key `"monomers"` [749].
4. Validate the JSON with an online validator or with `python -m json.tool molecules_config.json` [749].

##### Step 4.2 — Add config loading functions
Create a small helper module `phase1/config_loader.py` [749]:

```python
import json
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().parent / "data" / "molecules_config.json"

def load_config(role: str) -> list[dict]:
    """
    Load the molecule configuration for a given role.
    role: "PIs" or "monomers"
    """
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Missing configuration file: {CONFIG_FILE}")
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(role, [])

def save_config(role: str, molecules: list[dict]) -> None:
    """Persist the updated configuration back to JSON."""
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    data[role] = molecules
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def add_molecule(role: str, primary_names: list[str], alt_name: str | None, role_label: str) -> None:
    """
    Append a new molecule to the JSON config, avoiding duplicates.
    role: "PIs" or "monomers"
    role_label: "PI_TypeI", "PI_TypeII", "co-initiator", or "monomer"
    """
    molecules = load_config(role)
    existing = [n for m in molecules for n in m.get("primary_names", [])]
    if any(name in existing for name in primary_names):
        return
    molecules.append({
        "primary_names": primary_names,
        "alt_name": alt_name,
        "role": role_label
    })
    save_config(role, molecules)
```

##### Step 4.3 — Update the fetch scripts
In `fetch_molecules_PIs.py` [749]:
```python
from config_loader import load_config
molecules_config = load_config("PIs")
```
Remove the hardcoded list. The rest of the script remains unchanged [749].
Same for `fetch_molecules_monomers.py` [749]:
```python
from config_loader import load_config
molecules_config = load_config("monomers")
```

##### Step 4.4 — Expose a fetch_single function
At the end of each fetch script, add [749]:
```python
def fetch_single_PI(name: str) -> str | None:
    """
    Fetch a single PI by name.
    Order:
      1. Search molecules_PIs.csv
      2. Add name to config if not present
      3. Call get_smiles_robust([name])
      4. Append result to molecules_PIs.csv
    """
    import pandas as pd
    from config_loader import add_molecule

    # 1. CSV lookup
    if OUTPUT_PI_CSV.exists():
        df = pd.read_csv(OUTPUT_PI_CSV)
        match = df[df["name"].str.lower() == name.lower()]
        if not match.empty:
            return match.iloc[0]["smiles"]

    # 2. Add to config
    add_molecule("PIs", [name], None, "PI_TypeI")

    # 3. Fetch from PubChem
    smiles, used_name = get_smiles_robust([name], None)
    if smiles is None:
        return None

    # 4. Append to CSV
    new_row = pd.DataFrame([[used_name, smiles, "PI_TypeI"]],
                           columns=["name", "smiles", "role"])
    df = pd.read_csv(OUTPUT_PI_CSV) if OUTPUT_PI_CSV.exists() else pd.DataFrame(columns=["name", "smiles", "role"])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(OUTPUT_PI_CSV, index=False)

    return smiles
```

##### Step 4.5 — Protect the main blocks
Wrap the current top‑level execution in [749]:
```python
if __name__ == "__main__":
    # existing main loop that iterates over molecules_config
    ...
```

##### Step 4.6 — Update Phase 5
In `phase5/app/molecule_lookup.py` [749]:
```python
import sys
from pathlib import Path

P1_DIR = Path(__file__).resolve().parent.parent.parent / "phase1"
sys.path.append(str(P1_DIR))

from fetch_molecules_PIs import fetch_single_PI
from fetch_molecules_monomers import fetch_single_monomer

def get_smiles(name: str, role: str) -> str | None:
    if role == "PI":
        return fetch_single_PI(name)
    elif role == "monomer":
        return fetch_single_monomer(name)
    else:
        raise ValueError(f"Unknown role: {role}")
```
Then use `get_smiles` in `PredictorState.handle_prediction` [749].

---

#### 5. Handling Concurrency (Optional Advanced Step)
If Phase 5 is deployed as a multi‑user app, two simultaneous requests could write to `molecules_config.json` at the same time, corrupting the file [750]. To prevent this, use a file lock [750]:

```bash
pip install filelock
```

```python
from filelock import FileLock

LOCK_FILE = CONFIG_FILE.with_suffix(".lock")

def add_molecule(role, primary_names, alt_name, role_label):
    with FileLock(str(LOCK_FILE)):
        # ... existing logic ...
```

---

#### 6. Alternatives Considered
| Approach | Why rejected (for now) |
| :--- | :--- |
| **SQLite database for molecules** | More setup, harder to inspect manually, overkill for <1000 molecules [750]. |
| **YAML configuration** | Requires an external dependency; JSON is sufficient [750]. |
| **Environment variables** | Unsuitable for structured data [750]. |
| **Remote API for molecules** | Adds a network dependency; a local file is more robust [750]. |

---

#### 7. Migration Checklist
- [ ] Create `molecules_config.json` with the current hardcoded lists [750].
- [ ] Create `config_loader.py` [750].
- [ ] Update `fetch_molecules_PIs.py` and `fetch_molecules_monomers.py` to load from JSON [750].
- [ ] Wrap their mains in `if __name__ == "__main__":` [750].
- [ ] Add `fetch_single_PI` and `fetch_single_monomer` functions [750].
- [ ] Update Phase 5 to call `get_smiles` [750].
- [ ] Test the full pipeline end‑to‑end with a new molecule (e.g., "Irgacure 379") [750].
- [ ] Commit both the JSON config and the refactored scripts [750].

---

#### 8. Expected Benefits
| Before | After |
| :--- | :--- |
| Add molecule → edit Python code | Add molecule → edit JSON file (or via app) [750] |
| Restart script to pick up changes | Automatically read on each invocation [750] |
| Cannot add molecules from Phase 5 | Phase 5 appends to JSON at runtime [750] |
| Logic and data mixed | Clear separation of concerns [750] |
| Git conflicts on Python files | Git conflicts limited to JSON (easier to merge) [750] |

---

#### 9. What NOT to Do
* Do **not** store SMILES in the JSON config [750]. The JSON is a list of names to fetch [750]. SMILES belong in the resulting CSV [750].
* Do **not** put sensitive data (API keys, credentials) in `molecules_config.json` [750]. Use a `.env` file for that [750].
* Do **not** edit the JSON programmatically without a lock in a multi‑user context [750].

---

#### 10. Conclusion
Externalizing the molecule configuration is a small, well‑scoped refactor with a large impact on maintainability and scalability [750]. It transforms the pipeline from a script with hardcoded data into a proper data‑driven system where the code and the data evolve independently [750].

This refactor is not required for the MVP to function, but it is the natural next step when the project moves from a demo to a real tool that grows over time [750].

Estimated effort: **2–4 hours**, including testing and validation [750].  
Recommended timeline: **after the MVP is submitted and validated**, before starting Phase 5 deployment at scale [750].
