# Phase 5 — Step 0b: Inference Module & Image Logic Extraction Development Log
### From Batch Scripts to a Runtime-Callable Inference Pipeline

#### Document Purpose
This document chronicles the extraction of the molecular image logic into a shared module and the construction of a runtime inference pipeline, both prerequisites for Phase 5 (the Reflex web application). It explains why the two refactors were necessary, how they were designed and executed, which software engineering and machine learning best practices were applied, and how correctness was validated through non-regression tests and timing measurements.

This log is the direct continuation of `phase5-shared-client-refactoring-log.md`. Where that document resolved the name → SMILES problem, this one resolves the SMILES → prediction problem.

---

#### 1. The Trigger: Why Phase 5 Needed a Second Refactor
The first refactor (Step 0) made the PubChem client safely importable and runtime-callable. That solved the first step of the Phase 5 pipeline:

```text
[User types "Benzophenone"] → [Name → SMILES resolution]   ✅ Step 0
```

But the remaining steps of the pipeline were still locked inside Phase 1 and Phase 2 batch scripts:

```text
[SMILES → Image]           ← Phase 1, batch-only
[Image → Embedding]        ← Phase 2, batch-only
[Features → Prediction]    ← Phase 4, model saved but no runtime loader
```

Each of these scripts executes its full workload at import time. A Reflex app cannot `from phase1.generate_images_PIs import smiles_to_grayscale` without triggering a complete batch that regenerates 208 images and writes them to disk. This is the same import-safety problem solved in Step 0 for the PubChem client, now occurring three more times.

The Phase 5 pipeline therefore required two complementary refactors:
1. Extract the image logic from Phase 1 into a shared module (this document, Sections 3–6).
2. Create a new runtime inference module that chains Phase 1 + 2 + 4 into a single callable function (this document, Section 7).

Both follow the same discipline as Step 0: classify before coding, extract the mechanism from the policy, validate with non-regression tests.

---

#### 2. The Training/Inference Preprocessing Contract
Before describing the refactor, it is worth articulating the principle that drove it:

> **Preprocessing is a contract between training and inference.**

At training time, Phase 1 produced $224 \times 224$ grayscale images from SMILES strings, and Phase 2 converted those images into 1280-dimensional embeddings. The XGBoost model was trained on those embeddings.

At inference time (Phase 5), the same transformation must be applied to the user's SMILES strings. If the two transformations differ in any detail — image size, grayscale formula, rotation convention, RGB channel replication — the embeddings fed to XGBoost will be off-distribution. The model will still produce a number, because XGBoost has no way to know that its input is unusual. But the number will be wrong, and there will be no error message.

This is the class of bugs that is invisible until someone with domain expertise notices the predictions are systematically off. For an MVP demonstration, catching it late is acceptable. For a production system, it is unacceptable. The cheap insurance is to make training-time and inference-time preprocessing share a single implementation, so divergence is impossible.

The refactor enforces that contract: `smiles_to_grayscale` exists in exactly one place, and every caller — Phase 1 batch, Phase 2 batch, Phase 5 inference — imports it.

---

#### 3. Classification of the Existing Code
Every block of `phase1/generate_images_PIs.py` was classified into one of three categories, mirroring Step 0's methodology:

| Category | Definition | Destination |
| :--- | :--- | :--- |
| **Shared logic** | Identical for every molecule family and every usage context | `shared/molecule_images.py` |
| **Family-specific data** | Paths, filenames, or molecule lists | Stays in `generate_images_*.py` |
| **File structure** | `main()`, CLI entry point, reporting | Stays in `generate_images_*.py` |

##### The classification result:

| Original block | Category | Rationale |
| :--- | :--- | :--- |
| `IMG_SIZE = (224, 224)` | Shared | Same image size everywhere |
| `smiles_to_grayscale(smiles)` | Shared | Pure SMILES → array transformation |
| `augment_rotations(image)` | Shared | Pure array → array transformation |
| `process_molecule_row(row)` | Shared | Same per-row orchestration for PI and monomers |
| `show_preview(images, meta, n=4)` | Shared (parameterized) | Same plot layout; output path differs per family |
| `BASE_DIR`, `DATA_DIR`, `IMAGES_DIR` | Family-specific | Different folder layout possible per family |
| `INPUT_CSV`, `OUTPUT_NPZ`, `OUTPUT_META` | Family-specific | Different filenames |
| `main()` | File structure | Reporting + orchestration |

The one design change inside a shared function was in `show_preview`. The original reached for the module-level `DATA_DIR` to save the preview PNG. Once the function moves to `shared/`, that global is no longer reachable. The fix is standard dependency injection: promote the path to a function parameter.

```python
### Before (PI-specific):
def show_preview(images, meta, n=4):
    ...
    plt.savefig(DATA_DIR / "preview.png")

### After (shared, parameterized):
def show_preview(images, meta, output_path: Path, n: int = 4):
    ...
    plt.savefig(output_path)
```

The caller passes `DATA_DIR / "preview.png"` (PI) or `DATA_DIR / "preview_monomers.png"` (monomers). Same function, different destinations. This mirrors exactly the pattern used in Step 0 for `PubChemClient.__init__`, where `cache_file` and `output_csv` became parameters.

---

#### 4. Two Possible Strategies
Two ways to satisfy the training/inference preprocessing contract were considered:

| Strategy | Description | Cost | Decoupling |
| :--- | :--- | :--- | :--- |
| **A — Import from Phase 1** | `inference/pipeline.py` imports `smiles_to_grayscale` directly from `phase1.generate_images_PIs` | ~5 minutes | Inference depends on Phase 1 |
| **B — Extract to shared** | Move image logic to `shared/molecule_images.py`; both Phase 1 and inference import from there | ~30 minutes | Inference is independent of Phase 1 |

Strategy A was rejected because it couples Phase 5 to a Phase 1 batch script that is really an artifact of the training pipeline, not a library. If Phase 1 is ever removed, rewritten, or reorganized, the inference pipeline breaks.

Strategy B was chosen. It preserves the abstraction boundary: `inference/` depends on `shared/` (a proper library layer), not on `phase1/` (a batch script). Libraries depend on libraries; scripts depend on libraries; libraries never depend on scripts.

---

#### 5. Architecture of the Shared Module

##### 5.1 File Layout (After Both Refactors)
```text
project_root/
├── shared/
│   ├── __init__.py
│   ├── pubchem_client.py          ← name → SMILES (Step 0)
│   └── molecule_images.py         ← SMILES → image (Step 0b)
├── inference/
│   ├── __init__.py
│   └── pipeline.py                ← SMILES → % conversion (Step 0b)
├── phase1/
│   ├── fetch_molecules_PIs.py             ← thin wrapper (Step 0)
│   ├── fetch_molecules_monomers.py        ← thin wrapper (Step 0)
│   ├── generate_images_PIs.py             ← thin wrapper (Step 0b)
│   ├── generate_images_monomers.py        ← thin wrapper (Step 0b)
│   ├── exploratory/
│   │   ├── test_client.py
│   │   └── test_pipeline.py
│   └── data/
│       └── ...
├── phase2/
│   └── extract_embeddings.py      ← still batch-only (documented limitation)
├── phase3/
├── phase4/
│   └── xgboost_model.json
└── ...
```

Before both refactors: six files of ~300 LOC each, most logic duplicated.
After both refactors: two shared modules, two inference modules, four thin wrappers.

##### 5.2 shared/molecule_images.py — Public API
| Function | Purpose | Contract |
| :--- | :--- | :--- |
| `IMG_SIZE` | Image dimensions constant | `(224, 224)` |
| `smiles_to_grayscale(smiles, size=IMG_SIZE)` | SMILES → uint8 grayscale array | Raises `ValueError` for invalid SMILES |
| `augment_rotations(image)` | Return [90°, 180°, 270°] rotations | Input: $H \times W$ array; output: list of 3 arrays |
| `process_molecule_row(row)` | CSV row → 0 or 4 records | Skips invalid SMILES with a diagnostic |
| `show_preview(images, meta, output_path, n=4)` | Save preview PNG | Parameterized on output path |

The module has no class because the functions are pure: they hold no state and share no configuration.

##### 5.3 inference/pipeline.py — Public API
| Function | Purpose |
| :--- | :--- |
| `predict(pi_smiles, monomer_smiles, is_aqueous, logp, pi_concentration, uv_dose)` | End-to-end prediction: two SMILES + four features → % conversion |
| `smiles_to_grayscale` | Re-exported from `shared.molecule_images` (for API stability) |
| `image_to_embedding(image)` | Grayscale image → 1280-D embedding (via frozen MobileNetV2) |

The module is deliberately Reflex-free, TensorFlow-free at import time, and XGBoost-free at import time. It has no side effects and no global state beyond the two lazy singletons.

---

#### 6. The Lazy Singleton Pattern

##### 6.1 The Problem
Two heavy resources are needed at inference time:

| Resource | Load cost | Size |
| :--- | :--- | :--- |
| TensorFlow + MobileNetV2 | ~7–8 s | ~250 MB RAM |
| XGBoost model | ~0.5 s | ~1 MB |

If either is loaded at import time, the Reflex server takes 8 seconds to become ready and every restart is punishing. If either is loaded per request, every prediction takes 8 seconds and the user abandons the app.

##### 6.2 The Solution
```python
_mobilenet_model = None

def _get_mobilenet():
    global _mobilenet_model
    if _mobilenet_model is None:
        from tensorflow.keras.applications import MobileNetV2
        ...
        _mobilenet_model = Model(...)
    return _mobilenet_model
```

The first call pays the cost, assigns the result to the module global, and every subsequent call in the same process returns in $O(1)$.

##### 6.3 Proof
The test `test_pipeline.py` calls `predict()` three times in a single process:

```text
First call  (loads models): 33.13%  (8.308 s)
Second call (cached):       33.13%  (0.280 s)
Third call  (cached):       33.13%  (0.282 s)
```

A **30× speedup** on the second call, and full determinism (identical output on all three calls).

##### 6.4 Important Clarification — Singleton Scope
The singleton persists for the lifetime of a Python process, not across processes.

| Scenario | Singleton helps? |
| :--- | :--- |
| `python -m test_pipeline` run twice in a row (two processes) | ❌ No — each process starts fresh |
| Same script, `predict()` called twice (one process) | ✅ Yes — 30× speedup |
| Reflex server, first request after startup | ❌ No — pays the 8 s cost once |
| Reflex server, all later requests | ✅ Yes — sub-second responses |

For the Phase 5 web app, the process is long-lived (the server), so all user requests after the first benefit from the cache.

---

#### 7. Validation Strategy

##### 7.1 Non-Regression Testing for Image Extraction
The correctness criterion for extracting `smiles_to_grayscale` was byte-identical outputs. The procedure:

```text
1. Back up:    molecular_images*.npz     → _backup_before_image_refactor/
               molecular_metadata*.csv   → _backup_before_image_refactor/
2. Refactor:   move logic to shared/molecule_images.py
3. Rerun:      python -m phase1.generate_images_PIs
               python -m phase1.generate_images_monomers
4. Compare:    np.array_equal(new_npz, backup_npz)   → must be True
               fc.exe new_metadata.csv backup_metadata.csv → must report no differences
```

- **Why `np.array_equal` for NPZ**: NPZ is a compressed binary format. Two NPZ files containing identical arrays can differ byte-for-byte due to compression internals and embedded metadata. The decoded array comparison is authoritative.
- **Why `fc.exe` for CSV**: CSV is text, uncompressed, and stable. Byte comparison here is stricter and catches subtle changes (trailing newlines, BOMs, delimiters).

##### 7.2 Timing Test for the Inference Pipeline
The correctness of the lazy singleton is verified by timing:
- **First call**: ~8 s (loads models into memory).
- **Second call**: ~0.3 s (proves caching works).
- **Third call**: ~0.3 s (confirms stability).

##### 7.3 Import Safety Verification
```powershell
python -c "import phase1.generate_images_PIs; import phase1.generate_images_monomers; import shared.molecule_images; import inference.pipeline; print('OK')"
```
Output `OK` with no batch activity and no TensorFlow delay confirms import safety and lazy loading.

---

#### 8. Obstacles Encountered and Resolutions

| Obstacle | Root cause | Resolution |
| :--- | :--- | :--- |
| `SyntaxError: invalid syntax` at `print(...)/` | Stray `/` character at end of line | Deleted trailing character |
| Pyright warning: signature mismatch for `show_preview` | Old local definition shadowing imported one | Deleted leftover local `show_preview` |
| `ModuleNotFoundError: No module named 'inference'` | `sys.path[0]` was script folder, not project root | Ran via `python -m phase1.exploratory.test_pipeline` from root |
| `ImportError: cannot import name 'predict'` | `inference/pipeline.py` was incomplete | Appended `image_to_embedding` and `predict` |
| `FileNotFoundError: xgboost_model.json not found` | Mismatched directory path assumption | Updated `XGBOOST_MODEL_PATH` to match actual save location |
| CSV non-regression reported "1578 differing bytes" | Backup file had two stray leading spaces on header line | Stripped prefix and verified identical content |
| PowerShell `copy/y` not recognized | PowerShell uses `Copy-Item -Force` | Used `Copy-Item -Force` or `cp -Force` |
| Both runs of `test_pipeline.py` took ~8 s | Each invocation starts a new process | Called `predict()` three times within one process to verify caching |

---

#### 9. Final State

##### 9.1 Quantitative Summary
| Metric | Before Step 0b | After Step 0b |
| :--- | :--- | :--- |
| Files containing SMILES → image logic | 3 (PI, monomer, inference) | 1 (`shared/molecule_images.py`) |
| Total LOC of image logic | ~450 | ~150 |
| Phase 1 batch scripts | Full logic, ~300 LOC each | Thin wrappers, ~90 LOC each |
| `inference/` depends on `phase1/` | Yes (if Strategy A) | No |
| Runtime-callable prediction | No | Yes (`predict()`) |
| Import-safe Phase 1 scripts | Yes | Yes |
| Import-safe inference module | N/A | Yes |

##### 9.2 Behavioral Summary
- Both `molecular_images*.npz` files are array-identical to pre-refactor versions.
- Both `molecular_metadata*.csv` files are byte-identical to pre-refactor versions.
- Phase 1 batch scripts are still runnable via `python -m phase1.generate_images_*`.
- `inference.pipeline.predict` returns deterministic, chemically plausible values (33.13% for Benzophenone + Acrylic acid at 150 mJ/cm²).
- The lazy singleton delivers a **30× speedup** on the second prediction in the same process.

##### 9.3 Phase 5 Readiness
The Reflex `PredictorState` can now be written cleanly:

```python
from phase1.fetch_molecules_PIs import PI_CLIENT
from phase1.fetch_molecules_monomers import MONO_CLIENT
from inference.pipeline import predict

class PredictorState(rx.State):
    @rx.event
    async def handle_prediction(self, form_data: dict):
        # Step 1: resolve names → SMILES
        pi_entry = PI_CLIENT.fetch_single_molecule(form_data["pi_name"], role="PI_TypeI")
        mono_entry = MONO_CLIENT.fetch_single_molecule(form_data["monomer_name"], role="monomer")
        
        if pi_entry is None or mono_entry is None:
            self.error_message = "Molecule not found"
            return

        # Step 2: run the pipeline (handles Phase 1 → 2 → 4 internally)
        self.predicted_conversion = predict(
            pi_smiles=pi_entry["smiles"],
            monomer_smiles=mono_entry["smiles"],
            is_aqueous=1 if form_data["environment"] == "Aqueous" else 0,
            logp=form_data["logp"],
            pi_concentration=form_data["pi_concentration"],
            uv_dose=form_data["uv_dose"],
        )
```

---

#### 10. Lessons Learned

1. **Preprocessing Is a Contract, Not Just a Function**: The inference code must work on the exact same distribution the model was trained on. Sharing physical implementation guarantees zero train/inference drift.
2. **Extract when the Cost of Duplication Exceeds the Cost of Refactoring**: Strategy B required 30 minutes of restructuring but decoupled inference from batch training scripts permanently.
3. **Lazy Loading Is Not Optional for ML in Web Apps**: Loading a 250 MB TensorFlow model per request is unusable. Lazy singletons pay the 8 s load cost once on process startup.
4. **Timing and Byte-Comparison Are Complementary Tests**: Deterministic text uses byte comparisons, decoded arrays use array equality, and performance optimizations use timing measurements.
5. **Paths Are Contracts**: Centralizing model and file paths prevents subtle `FileNotFoundError` mismatches across pipeline stages.
6. **In-Process vs. Cross-Process State**: Singletons and in-memory caches persist across function calls within a process, not across independent script invocations.
7. **The Same Discipline Scales**: The classify → extract → thin wrapper → validate → document workflow is general and repeatable.

---

#### 11. Conclusion
The extraction of image logic into `shared/molecule_images.py` and the construction of `inference/pipeline.py` complete the second half of Phase 5's prerequisites. The pipeline now has four clean, decoupled layers:

| Layer | Responsibility | Module |
| :--- | :--- | :--- |
| **Resolution** | Name → SMILES | `shared/pubchem_client.py` |
| **Preprocessing** | SMILES → image | `shared/molecule_images.py` |
| **Inference** | SMILES + features → prediction | `inference/pipeline.py` |
| **Application** | UI + state orchestration | `app/` |

Each layer is independently testable, import-safe, and free of side effects at import time. The Reflex app can now be written as a thin orchestration layer connecting user inputs to a reactive UI.
