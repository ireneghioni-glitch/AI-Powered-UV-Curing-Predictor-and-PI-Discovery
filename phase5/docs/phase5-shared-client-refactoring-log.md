# Phase 5 — Step 0: Shared Client Refactoring Development Log
## From Duplicated Scripts to a Cohesive Client Architecture

### Document Purpose
This document chronicles the refactoring of the Phase 1 molecule-fetching scripts into a shared, reusable client, a prerequisite for Phase 5 (the Reflex web application). It explains why the refactor was necessary, how it was designed and executed, which software engineering best practices were applied, and how the correctness of the refactor was validated through unit tests and non-regression checks.

This log is intended both as documentation and as a learning resource on safe refactoring of a working codebase.

---

### 1. The Trigger: Why Phase 5 Forced a Refactor

#### 1.1 The MVP Specification Context
The MVP Technical Specification (`curing-prediction-pipeline-mvp-v3-en.md`) defines Phase 5 as an interactive Reflex web application where a user types a molecule name and receives a curing-conversion prediction. The predicted flow is:

```text
[User types: "Benzophenone"]
│
▼
[Name → SMILES resolution]        ← Phase 1 logic, called at runtime
│
▼
[SMILES → Image]                  ← Phase 1 (RDKit)
│
▼
[Image → Embedding]               ← Phase 2 (MobileNetV2)
│
▼
[Features → XGBoost prediction]   ← Phase 4
│
▼
[Result returned to the browser]
```

The critical observation: the first step (name → SMILES) is exactly what `fetch_molecules_PIs.py` and `fetch_molecules_monomers.py` already do. But those scripts were written for batch execution during Phase 1, not for runtime lookup inside a web server. Three concrete problems emerged:

| Problem | Why it breaks Phase 5 |
| :--- | :--- |
| `fetch_single_PI` was hardcoded with `role="PI_TypeI"` | Wrong role for monomers, co-initiators, or Type II PIs |
| The two scripts duplicated ~90% of their logic | Any bug fix or improvement had to be applied twice |
| Neither script was safe to import | Importing `fetch_molecules_PIs.py` executed the entire batch fetch, sending 56 HTTP requests |

The third point is the killer: in Phase 5, we need to do `from fetch_molecules_PIs import something` without any side effects. But the original file executed `main()` at module level, so any import would trigger a full network batch.

#### 1.2 The Realization
What we actually needed was not two scripts, but one client class with two configured instances:
* `PI_CLIENT` — configured with the PI manual dictionary and PI CSV paths
* `MONO_CLIENT` — configured with the monomer manual dictionary and monomer CSV paths

Both instances expose the same methods. Phase 5 imports the instances and calls methods. The batch scripts become thin wrappers that use the same instances.

This is a textbook application of the Dependency Injection and Single Responsibility principles: the mechanism (how to fetch a SMILES from PubChem) is separated from the policy (which molecules we care about).

---

### 2. Classification of the Existing Code

Before writing any new code, every block of the original `fetch_molecules_PIs.py` was classified into one of three categories:

| Category | Definition | Destination |
| :--- | :--- | :--- |
| **Shared logic** | Identical for every molecule family | `shared/pubchem_client.py` |
| **Family-specific data** | Depends on PI vs monomer | Stays in `fetch_*.py` |
| **File structure** | Imports, main, reporting | Stays in `fetch_*.py` (rewritten) |

The result of the classification:

| Original block | Category | Rationale |
| :--- | :--- | :--- |
| `_CSV_LOCK` | Shared | Same lock needed for every CSV |
| `_load_csv_safe` | Shared | Pure function on path |
| `_append_molecule` | Shared | Pure function on path |
| `load_cache` / `save_cache` / `_save_to_cache` | Shared | Same logic, different file |
| `get_cid_from_name` | Shared | Pure PubChem REST call |
| `get_smiles_from_cid` | Shared | Pure PubChem REST call |
| `get_smiles_robust` | Shared | Orchestration, identical logic |
| `MANUAL_SMILES` (dict) | Family-specific | PI names ≠ monomer names |
| `OUTPUT_PI_CSV` / `CACHE_FILE` | Family-specific | Different paths per family |
| `molecules_config` | Family-specific | List of PI molecules |
| `fetch_single_PI` | Shared (generalized) | Runtime entry point |
| `main()` | File structure | Reporting + loop |

The rule that drove every decision:
> If the logic depends on a value (path, dictionary) but the algorithm around it is identical, then the logic is shared and the value becomes a constructor parameter.

This is the essence of the Dependency Injection pattern.

---

### 3. Architecture of the Shared Client

#### 3.1 File Layout

```text
project_root/
├── shared/
│   ├── __init__.py
│   └── pubchem_client.py          ← all shared logic (~250 LOC)
├── phase1/
│   ├── fetch_molecules_PIs.py     ← PI data + config + main (~180 LOC)
│   ├── fetch_molecules_monomers.py ← monomer data + config + main (~150 LOC)
│   ├── exploratory/
│   │   └── test_client.py         ← validation suite (~80 LOC)
│   └── data/
│       ├── molecules_PIs.csv
│       ├── molecules_monomers.csv
│       ├── smiles_cache_PIs.csv
│       └── smiles_cache_monomers.csv
```

Before the refactor: two files of ~330 LOC each, ~90% duplicated. After the refactor: one shared client + two thin wrappers, each with only family-specific data.

#### 3.2 The PubChemClient Class

```python
class PubChemClient:
    _PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"

    def __init__(
        self,
        manual_smiles: dict[str, str],
        cache_file: Path,
        output_csv: Path,
        request_timeout: float = 10.0,
    ) -> None:
        self.manual_smiles = dict(manual_smiles)
        self.manual_smiles_lower = {k.lower(): (k, v) for k, v in manual_smiles.items()}
        self.cache_file = Path(cache_file)
        self.output_csv = Path(output_csv)
        self.request_timeout = request_timeout
```

The four constructor parameters are the complete answer to the question "what does this client need to know about the molecule family it serves?". Everything else is shared.

#### 3.3 Public API
Two entry points, one for each usage pattern:

| Method | Usage | Called by |
| :--- | :--- | :--- |
| `get_smiles_robust(primary_names, alt_name)` | Batch resolution with fallback chain | Phase 1 `main()` |
| `fetch_single_molecule(name, role, csv_path)` | Runtime lookup for one molecule | Phase 5 Reflex State |

Both share the same three-tier resolution:

```text
1. Cache CSV        (no network)
2. Manual dictionary (no network)
3. PubChem REST     (name → CID → CanonicalSMILES)
```

---

### 4. Best Practices Applied

#### 4.1 DRY — Don't Repeat Yourself
The original two scripts had `get_smiles_robust` duplicated with only the dictionary and paths differing. Any bug fix required two edits. After the refactor, the algorithm lives in one place. Adding a third family (e.g. oligomers) in the future means constructing a third client instance — no logic duplication.

#### 4.2 Single Responsibility Principle
Each file has one job:
* `pubchem_client.py` — how to resolve names to SMILES
* `fetch_molecules_PIs.py` — which PI molecules we care about
* `fetch_molecules_monomers.py` — which monomer molecules we care about
* `test_client.py` — how to verify the client works

Before the refactor, `fetch_molecules_PIs.py` did four jobs simultaneously: data, algorithm, batch orchestration, and reporting.

#### 4.3 Dependency Injection via Constructor
The client does not reach out for global constants. It receives them:

```python
PI_CLIENT = PubChemClient(
    manual_smiles=MANUAL_SMILES,
    cache_file=CACHE_FILE,
    output_csv=OUTPUT_PI_CSV,
)
```

This makes the client testable in isolation (the test injects fake paths and a small dictionary), reusable across families, and immune to import-order issues (no hidden dependency on module-level state).

#### 4.4 Thread Safety — One Global Lock
All CSV writes are serialized through a single module-level lock:

```python
_CSV_LOCK = threading.Lock()
```

Why global and not per-instance? Because Reflex can run multiple requests concurrently, and two `fetch_single_molecule` calls could otherwise interleave:

```python
# Thread A                    # Thread B
df = _load_csv_safe(path)     df = _load_csv_safe(path)
# ... modifies df ...         # ... modifies df ...
df.to_csv(path)               df.to_csv(path)   # ← Thread B overwrites A's write
```

With a single lock shared across all instances, writes are atomic. The lock is coarse, but for our traffic (a few writes per second at most) the contention is negligible and the correctness is total.

#### 4.5 Defensive Programming — Safe CSV I/O
The original code assumed the CSV always existed and was well-formed. In a web server, this is wrong: the CSV may not exist on first run, may be mid-write from another process, or may be truncated by a crash.

The helper `_load_csv_safe` handles all these cases:

```python
def _load_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=_CSV_COLUMNS)
    try:
        df = pd.read_csv(path)
        return df if not df.empty else pd.DataFrame(columns=_CSV_COLUMNS)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=_CSV_COLUMNS)
```

Three failure modes covered: missing file, empty file, malformed file. The caller always receives a valid DataFrame.

#### 4.6 Idempotency — Deduplication on Append
A naïve `df.to_csv(..., mode='a')` would grow the CSV by one row on every call. In a web app where the same popular molecule (e.g. Benzophenone) is queried a hundred times, the file would balloon.

The `_append_molecule` helper checks for a case-insensitive match before writing:

```python
if not df.empty and (df["name"].astype(str).str.lower() == name.lower()).any():
    return  # already there
```

This makes `fetch_single_molecule` idempotent: calling it twice with the same name produces the same state.

#### 4.7 Case-Insensitive Lookups
Users type `benzophenone`, `Benzophenone`, `BENZOPHENONE`. The client normalizes at every tier:
* A precomputed `_MANUAL_SMILES_LOWER` index gives O(1) lookup.
* The CSV match uses `.str.lower() == key.lower()`.
* Cache lookups use the same normalization.

Without this, "Benzophenone" and "benzophenone" would produce two separate CSV rows.

#### 4.8 Backward Compatibility — Legacy Wrappers
The refactor changes `fetch_single_PI`'s return type from `str` to `dict`. Any existing Phase 1 code that called it and expected a string would break. To avoid this, a three-line wrapper preserves the old signature:

```python
def fetch_single_PI(name: str) -> str | None:
    result = PI_CLIENT.fetch_single_molecule(name, role="PI_TypeI")
    return result["smiles"] if result else None
```

This is the Strangler Fig pattern: the new implementation is introduced alongside the old interface, and the old interface is kept as a shim until all callers migrate.

#### 4.9 Import Safety — `if __name__ == "__main__":`
The original file executed the batch fetch at module level. This is a latent bug for any library that imports it. The fix:

```python
if __name__ == "__main__":
    main()
```

Now `from phase1.fetch_molecules_PIs import PI_CLIENT` is a pure operation: it constructs the client, computes the lowercase index, and returns — no network, no CSV writes, no side effects.

This is arguably the single most important change for Phase 5: without it, the Reflex app would fire 56 HTTP requests to PubChem at import time.

#### 4.10 Forward-Compatible Type Hints
The type syntax `str | None` (PEP 604) requires Python 3.10+. The `chemvision` environment runs Python 3.9, where this syntax raises `TypeError` at class/function definition time. The fix is the module-level future import:

```python
from __future__ import annotations
```

This defers annotation evaluation to static analysis only, making the code compatible with Python 3.7+. It's a one-line cost for cross-version portability and it should be the first executable line of every file that uses modern type hints.

---

### 5. Validation Strategy

The refactor touched every line of two working scripts. A silent regression would be invisible until Phase 5. To prevent this, validation was performed in four independent layers.

#### 5.1 Layer 1 — Unit Tests on the Client (Checkpoints)
Before any wrapper was written, a disposable `test_client.py` exercised the client in isolation with a temporary directory (via `tempfile.mkdtemp`). Nine checkpoints covered every branch:

| Checkpoint | What it proves |
| :--- | :--- |
| 1 | `_append_molecule` dedup: Lock + case-insensitive dedup work |
| 2 | `_load_csv_safe` on missing file: No crash on first run |
| 3 | Client construction: Lowercase index is populated |
| 4 | `fetch_single_molecule` via manual dict: Offline path works, role respected |
| 5 | Idempotency: No duplicate rows on repeated calls |
| 6 | `fetch_single_molecule` via PubChem API: Live network path works |
| 6b | Role persistence: CSV value takes precedence over parameter |
| 7 | Unknown molecule: Returns `None`, never raises |
| 8 | `get_smiles_robust` manual path: Batch entry point works |
| 9 | `get_smiles_robust` `alt_name` fallback: Tier-3 fallback works |

The suite ran in a temporary directory that was destroyed on exit, so it left no artifacts and could be run repeatedly without state contamination.

#### 5.2 Layer 2 — Non-Regression Testing (Backup + Compare)
For the batch wrappers, the correctness criterion was stronger than "the code runs": the output must be byte-identical to the pre-refactor output. The procedure:

```text
1. Copy the current CSV:   molecules_PIs.csv → molecules_PIs.csv.bak
2. Apply the refactor.
3. Run the new wrapper:    python -m phase1.fetch_molecules_PIs
4. Compare:                fc.exe molecules_PIs.csv molecules_PIs.csv.bak
```

The Windows `fc.exe` tool returns "FC: no differences encountered" if the files are byte-identical. This is a perfect non-regression test: it catches any change in SMILES resolution, molecule ordering, column structure, or delimiter behavior.

The same procedure was applied to the monomer wrapper. Both produced identical output.

Why a byte comparison and not a spot-check? Because regressions are often subtle: a trailing newline added by a different pandas version, a column reordered, a UTF-8 BOM added. A spot-check of three rows would miss all of these. Byte comparison catches everything.

#### 5.3 Layer 3 — Import Safety Verification
After both wrappers were refactored, the following command was executed from the project root:

```powershell
python -c "from phase1.fetch_molecules_PIs import PI_CLIENT; from phase1.fetch_molecules_monomers import MONO_CLIENT; print('OK')"
```

The output was `OK`, with no network activity. This confirms four properties simultaneously:
1. `if __name__ == "__main__":` is in place in both files.
2. `from __future__ import annotations` is present in both.
3. Both `PubChemClient` instances are constructible independently.
4. No name collisions between the two modules (`PI_CLIENT` vs `MONO_CLIENT`, both `MANUAL_SMILES` dicts are properly scoped to their modules).

This is the exact import pattern Phase 5 will use. Verifying it before Phase 5 starts means the integration step cannot fail for environmental reasons.

#### 5.4 Layer 4 — Legacy Behavior Preservation
The legacy wrapper `fetch_single_PI` was tested to still return a plain SMILES string or `None`. This was implicit in the fact that no other Phase 1 code raised an `AttributeError`, but the contract is preserved and documented in its docstring.

---

### 6. Obstacles Encountered and Resolutions

| Obstacle | Root cause | Resolution |
| :--- | :--- | :--- |
| `FileNotFoundError: '\tmp\pubchem_test'` | POSIX path `/tmp` doesn't exist on Windows | Switched to `tempfile.mkdtemp()`, which is cross-platform |
| `ModuleNotFoundError: shared` when running `python phase1/test_client.py` | `sys.path[0]` is the script's directory, not the project root | Ran via `python -m phase1.test_client` from the root instead |
| Test 6 assertion failed: `role="test"` instead of `"monomer"` | The test itself appended "Aspirin" with `role="test"` at checkpoint 1, then expected the client to overwrite it at checkpoint 6 | Recognized that this was a design decision, not a bug: the CSV value is authoritative once a molecule is catalogued. Added checkpoint 6b to document the persistence rule explicitly |
| `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` | `str | None` (PEP 604) requires Python 3.10+; the env runs 3.9 | Added `from __future__ import annotations` as the first executable line |

Each obstacle reinforced a general lesson:
* **Obstacle 1:** never hardcode OS-specific paths; use `tempfile`, `pathlib`, or `os.path`.
* **Obstacle 2:** the `-m` flag is not optional for multi-package projects; it is the correct way to invoke modules.
* **Obstacle 3:** a failing test is not always a bug. Sometimes it reveals an undocumented design decision. Distinguishing the two requires asking "is the code wrong, or is the expectation wrong?" before "fixing" anything.
* **Obstacle 4:** `from __future__ import annotations` is the cheapest insurance against version-dependent type-hint failures.

---

### 7. Final State

#### 7.1 Quantitative Summary

| Metric | Before | After |
| :--- | :--- | :--- |
| Files with fetch logic | 2 | 1 shared + 2 wrappers |
| Total lines of fetch code | ~660 | ~250 shared + ~330 data |
| Duplicated logic | ~90% | 0% |
| Safe to import | No | Yes (both wrappers) |
| Runtime-callable | No (batch only) | Yes (`fetch_single_molecule`) |
| Thread-safe | No | Yes (global lock) |
| Idempotent | No | Yes (dedup on append) |
| Case-insensitive | Partial | Complete |
| Version-compatible | Python 3.10+ | Python 3.7+ |

#### 7.2 Behavioral Summary
* Both CSVs are byte-identical to the pre-refactor versions.
* Both wrappers are importable without side effects.
* Both wrappers run as batch commands via `python -m phase1.fetch_molecules_*`.
* The client passes all 9 + 1 checkpoints.
* The legacy `fetch_single_PI` wrapper still returns a string for backward compatibility.

#### 7.3 Phase 5 Readiness
The Phase 5 `PredictorState` can now be written as:

```python
from phase1.fetch_molecules_PIs import PI_CLIENT
from phase1.fetch_molecules_monomers import MONO_CLIENT

class PredictorState(rx.State):
    @rx.event
    async def handle_prediction(self, form_data: dict):
        pi_entry = PI_CLIENT.fetch_single_molecule(form_data["pi_name"], role="PI_TypeI")
        mono_entry = MONO_CLIENT.fetch_single_molecule(form_data["monomer_name"], role="monomer")
        if pi_entry is None or mono_entry is None:
            self.error_message = "Molecule not found"
            return
        # pi_entry["smiles"], mono_entry["smiles"] are ready for the RDKit pipeline
```

No conditional logic on the family type, no manual CSV handling, no risk of a network call triggering on import.

---

### 8. Lessons Learned

#### A. Refactor Before Adding, Not After Breaking
The refactor was triggered by Phase 5's needs, but it could have been done at any point after Phase 1. Doing it before Phase 5 was written means Phase 5 starts from a clean foundation instead of inheriting two duplicated scripts and accumulating technical debt.

#### B. Classify Before Coding
The classification exercise (Section 2) took ten minutes and saved hours. Writing code before deciding what is shared and what is family-specific leads to accidental coupling that is painful to undo.

#### C. Tests Are the Contract
Nine checkpoints were written before the wrappers were refactored. This meant that by the time the wrappers were rewritten, any bug in the client would surface in isolation, before being obscured by wrapper behavior. Writing tests first also forced clarity about the public API.

#### D. Byte-Level Non-Regression Beats Spot-Checks
`fc.exe` is a blunt instrument, but for CSV output that should not change at all, it is perfect. It caught nothing in this refactor (both files matched), but its presence as a gatekeeper meant the refactor was executed with the knowledge that any silent change would be caught.

#### E. Backward Compatibility Is a Feature
Adding a three-line wrapper to preserve `fetch_single_PI` cost nothing and eliminated the risk of breaking unseen Phase 1 callers. The cost of backward compatibility is usually tiny compared to the cost of tracking down a broken caller in a codebase you haven't touched in months.

#### F. Every Failure Is an Opportunity to Document
Obstacle 3 (the role test failure) turned into checkpoint 6b, which is now executable documentation of a design rule that would otherwise live only in the developer's head. Failures are not interruptions; they are the moments when implicit rules become explicit.

#### G. The Environment Is Part of the System
Both environment-specific obstacles (`/tmp` on Windows, Python 3.9's lack of PEP 604) had nothing to do with the algorithm and everything to do with the runtime. A professional treats the environment as a first-class concern: paths via `tempfile`/`pathlib`, type hints via `from __future__ import annotations`, invocation via `python -m`.

---

### 9. Conclusion

The refactoring of the Phase 1 scripts into a shared client is a small, contained example of a discipline that distinguishes production code from exploratory notebooks:
> Extract the mechanism from the policy, test the mechanism in isolation, verify the policy against a known-good output, and document the design decisions as executable assertions.

The work described here took less than a day, produced no functional change in the existing Phase 1 outputs, and unblocked Phase 5 with a clean, tested, documented foundation. It is a template for future refactors in this project and, more broadly, a demonstration that architectural decisions made early — classification, dependency injection, import safety, thread safety, idempotency — pay dividends precisely when the system grows beyond its original scope.
