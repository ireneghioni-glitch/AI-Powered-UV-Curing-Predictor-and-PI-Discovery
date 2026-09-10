# Phase 5 — Complete Guide: From Integration to a Working Reflex App

We are ready for the final phase. Before writing any code, however, we need a careful survey of what is missing: the pipeline is complete in phases 1–4, but it is not yet callable at runtime from a web app. This guide bridges that gap methodically.

---

## 1. The Big Picture: What Phase 5 Actually Needs

### 1.1 The user flow vs the real flow
**What the end user sees:**
```text
[Types "Benzophenone" and "TMPTA"] -> [Clicks] -> [Sees "78.3%"]
```

**Behind the scenes, every click executes five steps:**

| Step | Resource required | Setup cost | When to load it |
| :--- | :--- | :--- | :--- |
| **1** | Name -> SMILES | `PI_CLIENT` / `MONO_CLIENT` | Already ready (Step 0) - Import |
| **2** | SMILES -> Image | RDKit (`Draw.MolToImage`) | ~0 (pure Python) - Per request |
| **3** | Image -> Embedding | MobileNetV2 (TensorFlow) | 3–5 s - Once only |
| **4** | Features -> Prediction | XGBoost model | 0.5 s - Once only |
| **5** | Save log | SQLite via SQLModel | ~0 - Per request |

> **The key insight:** Steps 3 and 4 are expensive to initialize. If you reloaded them on every click, each prediction would take 4–6 seconds and the user would abandon the app. The solution is the **lazy singleton pattern**: load once, on the first request, then reuse forever.

### 1.2 The architectural separation
The Reflex code must not contain ML logic. It should contain only:
* UI (visual components)
* State (reactive variables + event handlers)

All prediction logic (RDKit + MobileNetV2 + XGBoost) goes into a separate module, `inference/pipeline.py`, which:
* Is testable in isolation (without starting Reflex)
* Is reusable (future CLI, REST API, notebook)
* Isolates the "weight" of the models in a single place
* Has no Reflex dependency

This separation is the **Single Responsibility Principle** applied at project level: Reflex draws, the pipeline computes.

---

## 2. Theory: The Reflex Reactive Model

Before writing code, understanding what Reflex does under the hood changes how you use it.

### 2.1 Client <-> Server: WebSocket, not HTTP
A traditional web app (Flask, Django):
```text
[Browser] --HTTP POST--> [Server] --HTTP 200--> [Browser]
(request)                                 (HTML response)
```
On every click, the browser reloads the page. Simple but slow.

Reflex works like this:
```text
[Browser] <====WebSocket====> [Python Server]
            (persistent channel)
```
The browser keeps an open connection with the Python server. When the user clicks a button:
1. The browser sends a message over the WebSocket: `"call PredictorState.handle_prediction"`
2. The server runs the Python method
3. Every change to a state variable (`self.predicted_conversion = 78.3`) is translated into a JSON patch and sent back
4. The browser receives the patch and updates only the piece of UI that depends on that variable

This is the **reactive model**: you don't reload the page, you update only what changes.

### 2.2 State Vars and Event Handlers
```python
class PredictorState(rx.State):
    # Reactive variables: any change triggers an automatic re-render
    pi_name: str = ""
    predicted_conversion: float = 0.0
    is_loading: bool = False

    # Event handlers: methods that mutate the variables above
    def update_pi_name(self, value: str):
        self.pi_name = value
```
**The implicit contract:** Anything the UI reads must be a state var. If you try to read an instance variable that is not declared at class level, Reflex does not "see" it and will not react to its changes.

### 2.3 async + yield: The Pattern for Long Operations
The critical point: a call to MobileNetV2 blocks the Python thread for ~200 ms. Without `yield`, the browser would not see the spinner until the very end.

```python
@rx.event
async def handle_prediction(self, form_data: dict):
    self.is_loading = True
    yield  # sends state to the browser NOW (spinner visible)

    # ... 200 ms of heavy computation ...

    self.predicted_conversion = 78.3
    self.is_loading = False
    yield  # sends the result to the browser
```
Each `yield` is a checkpoint: *"this is the current state, send it to the client before continuing"*. Without them, the user sees only the final result, never the loading indicator.

### 2.4 SQLModel vs Pure SQLAlchemy
`rx.Model` is a wrapper around SQLModel (which is itself SQLAlchemy + Pydantic). Three advantages over pure SQLAlchemy:
1. **Python classes as schema:** the table is declared as a class, not with `CREATE TABLE`
2. **Automatic validation:** types are checked by Pydantic
3. **Automatic migrations:** `reflex db make-migrations` generates the Alembic files

```python
class CuringLog(rx.Model, table=True):   # table=True -> generates a table
    pi_smiles: str                        # -> column TEXT NOT NULL
    uv_dose: float                        # -> column REAL NOT NULL
    created_at: datetime = ...            # -> column DATETIME DEFAULT now
```

---

## 3. Step 1: Prepare the Inference Pipeline

This is the most important step and one that no Reflex tutorial will show you, because it is specific to your project.

### 3.1 What Is Missing
Look at your current files:

| File | What it does | Problem for Phase 5 |
| :--- | :--- | :--- |
| `generate_images_PIs.py` | Batch: reads CSV -> writes NPZ | Runs everything at import |
| `phase2/extract_embeddings.py` | Batch: reads NPZ -> writes NPY | Runs everything at import |
| `phase3/model.py` | Defines the `CuringPredictorNet` class | Importable ✅ but not needed (we use XGBoost) |
| `phase4/train_xgboost.py` | Batch: trains and saves the model | The saved model is fine, but the code has no loader |

None of these files is callable at runtime. They need a wrapper that exposes a single function:

```python
predicted_conversion = predict(
    pi_smiles="O=C(C1=CC=CC=C1)C2=CC=CC=C2",
    monomer_smiles="C=C(C)C(=O)OCC...",
    is_aqueous=0,
    logp=2.5,
    pi_concentration=2.0,
    uv_dose=150.0,
)
```

### 3.2 The Lazy Singleton Pattern
The problem to solve:

```python
# ❌ WRONG: reloads the model on every prediction
def predict(...):
    model = xgb.XGBRegressor()
    model.load_model("xgboost_model.json")   # 500 ms wasted
    cnn = MobileNetV2(weights='imagenet')    # 4 s wasted
    ...

# ❌ WRONG: loads at import time (blocks startup)
model = xgb.XGBRegressor()
model.load_model("xgboost_model.json")
cnn = MobileNetV2(weights='imagenet')
```

The solution:
```python
# ✅ RIGHT: loads on first call, reuses afterwards
_model = None

def _get_model():
    global _model
    if _model is None:
        _model = xgb.XGBRegressor()
        _model.load_model("xgboost_model.json")
    return _model
```
The cost is paid only once, on the first prediction. Subsequent ones are instantaneous (milliseconds).
The same pattern applies to MobileNetV2.

### 3.3 The `inference/pipeline.py` Module
Create the folder `inference/` at the project root with two files:
```text
project_root/
├── inference/
│   ├── __init__.py
│   └── pipeline.py
```

`inference/pipeline.py`:

```python
"""
Runtime inference pipeline for the UV-curing predictor.

Combines four phases into a single callable:
Phase 1: SMILES -> grayscale image (RDKit + OpenCV)
Phase 2: image -> embedding (MobileNetV2, frozen)
Phase 4: [embedding_PI | embedding_monomer | env_features] -> XGBoost

Heavy resources (TensorFlow MobileNetV2, XGBoost model) are loaded lazily on the first
call and cached as module-level singletons. This is critical for the Reflex web app:
loading them at import time would block the app startup by ~4 seconds and loading them
per request would make every prediction take 4+ seconds.

This module is deliberately free of any Reflex dependency, so it can be:
* unit-tested standalone
* reused by a future CLI or REST API
* imported by a notebook without side effects
"""

from __future__ import annotations
import os
from pathlib import Path

# Silence TensorFlow oneDNN warnings BEFORE importing tensorflow.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import cv2
import numpy as np
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Draw

# ==================== PATHS ====================
BASE_DIR = Path(__file__).resolve().parent.parent
PHASE4_DATA = BASE_DIR / "phase4" / "data"
XGBOOST_MODEL_PATH = PHASE4_DATA / "xgboost_model.json"

# ==================== CONSTANTS ====================
IMG_SIZE = (224, 224)

# ==================== LAZY SINGLETONS ====================
_mobilenet_model = None    # set by _get_mobilenet()
_xgboost_model = None      # set by _get_xgboost()


def _get_mobilenet():
    """
    Lazily load the frozen MobileNetV2 embedding extractor.

    The first call imports TensorFlow (~3-4 s) and builds the model.
    Subsequent calls return the cached instance in O(1).
    """
    global _mobilenet_model
    if _mobilenet_model is None:
        from tensorflow.keras import Model
        from tensorflow.keras.applications import MobileNetV2
        from tensorflow.keras.layers import GlobalAveragePooling2D

        base = MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
        )
        base.trainable = False
        gap = GlobalAveragePooling2D()(base.output)
        _mobilenet_model = Model(inputs=base.input, outputs=gap)
    return _mobilenet_model


def _get_xgboost():
    """Lazily load the trained XGBoost model from disk."""
    global _xgboost_model
    if _xgboost_model is None:
        if not XGBOOST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"XGBoost model not found at {XGBOOST_MODEL_PATH}. "
                f"Run phase4/train_xgboost.py first."
            )
        _xgboost_model = xgb.XGBRegressor()
        _xgboost_model.load_model(str(XGBOOST_MODEL_PATH))
    return _xgboost_model


# ==================== PIPELINE STEPS ====================
def smiles_to_grayscale(smiles: str, size: tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """
    Convert a SMILES string to a 224x224 grayscale image (uint8).

    Mirrors the Phase 1 logic: RDKit draws the molecule, OpenCV converts
    to grayscale. Raises ValueError for invalid SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    img = Draw.MolToImage(mol, size=size)
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    return gray


def image_to_embedding(image: np.ndarray) -> np.ndarray:
    """
    Pass a grayscale image through frozen MobileNetV2.

    Returns a 1-D array of shape (1280,).
    """
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    # Grayscale -> RGB by channel replication (matches Phase 2)
    rgb = np.stack([image] * 3, axis=-1).astype(np.float32)   # (224,224,3)
    batch = np.expand_dims(rgb, axis=0)                       # (1,224,224,3)
    batch = preprocess_input(batch)
    embedding = _get_mobilenet().predict(batch, verbose=0)    # (1,1280)
    return embedding[0]


# ==================== PUBLIC ENTRY POINT ====================
def predict(
    pi_smiles: str,
    monomer_smiles: str,
    is_aqueous: int,
    logp: float,
    pi_concentration: float,
    uv_dose: float,
) -> float:
    """
    End-to-end prediction: two SMILES + 4 environmental features -> % conversion.

    Parameters
    ----------
    pi_smiles, monomer_smiles : str
        Valid SMILES strings, already resolved from names.
    is_aqueous : int
        0 for solvent, 1 for aqueous.
    logp : float
        Octanol-water partition coefficient of the photoinitiator.
    pi_concentration : float
        Photoinitiator concentration in percent (typically 1–5).
    uv_dose : float
        UV energy dose in mJ/cm² (typically 50–500).

    Returns
    -------
    float
        Predicted double-bond conversion in percent, clipped to [0, 100].
    """
    # Phase 1 -> 2: SMILES -> embedding (1280-D each)
    pi_img = smiles_to_grayscale(pi_smiles)
    mono_img = smiles_to_grayscale(monomer_smiles)
    pi_emb = image_to_embedding(pi_img)
    mono_emb = image_to_embedding(mono_img)

    # Phase 4: concatenate [1280 | 1280 | 4] = 2564 features
    features = np.concatenate([
        pi_emb, mono_emb,
        [is_aqueous, logp, pi_concentration, uv_dose],
    ]).reshape(1, -1)

    # Phase 4: predict
    raw = float(_get_xgboost().predict(features)[0])
    return float(np.clip(raw, 0.0, 100.0))
```

**Key points to notice:**
* `os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")` must be before any `import tensorflow`. That is why it is at the top of the file.
* `_get_mobilenet` imports TensorFlow inside the function. This delays the 3–4 second TF import until the first prediction, instead of blocking app startup.
* The `predict` function is pure: it takes SMILES and features, returns a float. It does not touch Reflex, does not touch the DB. It is testable in isolation.
* `np.clip(raw, 0, 100)` guarantees that an overconfident model never returns absurd values.

### 3.4 Testing the Pipeline (Isolated)
Before writing a single line of Reflex, verify that the pipeline works:

```python
# test_pipeline.py (temporary, at the project root)
from inference.pipeline import predict

result = predict(
    pi_smiles="O=C(C1=CC=CC=C1)C2=CC=CC=C2",       # Benzophenone
    monomer_smiles="C=CC(=O)O",                     # Acrylic acid
    is_aqueous=0,
    logp=3.0,
    pi_concentration=2.0,
    uv_dose=150.0,
)
print(f"Predicted conversion: {result:.2f}%")
```
The first run will take 4–6 seconds (TensorFlow + XGBoost loading). The second, if you time it, will be ~200 ms. This is the payoff of the lazy singleton.

If this test works, the pipeline is ready. If it does not work, do not move to Reflex. Debugging a TensorFlow issue inside a Reflex handler is 10 times harder than debugging it in a script.

---

## 4. Step 2: Initialize the Reflex Project

From the project root:
```powershell
reflex init
```
Choose `0` (Blank template). Reflex creates this structure:
```text
AI-Powered-UV-Curing-Predictor-and-PI-Discovery/
├── rxconfig.py
├── AI_Powered_UV_Curing_Predictor_and_PI_Discovery/   <- auto-generated name
│   ├── __init__.py
│   └── AI_Powered_UV_Curing_Predictor_and_PI_Discovery.py
└── ... (your existing folders: shared/, phase1/, ..., inference/)
```

Rename the app folder to `app/` (cleaner and more stable over time) and update `rxconfig.py`:

```python
# rxconfig.py
import reflex as rx

config = rx.Config(
    app_name="app",   # <- the name of the app folder
)
```
Also rename the internal file `AI_Powered_...py` to `app/app.py`.

**Recommended final structure:**
```text
AI-Powered-UV-Curing-Predictor-and-PI-Discovery/
├── rxconfig.py
├── app/
│   ├── __init__.py
│   ├── app.py           <- entry point: rx.App() + add_page
│   ├── models.py        <- CuringLog
│   ├── state.py         <- PredictorState
│   └── pages/
│       ├── __init__.py
│       └── index.py     <- home page UI
├── shared/
├── phase1/
├── phase2/
├── phase3/
├── phase4/
└── inference/
```
**The criterion:** The Reflex app lives in `app/`, everything else is "logical backend" that the app imports. This separation prevents the app from becoming a monolithic blob.

---

## 5. Step 3: Define the Database Model

### 5.1 Theory: What to Save and Why
A prediction app produces three types of information:

| Type | Example | Should it be saved? |
| :--- | :--- | :--- |
| **User input** | SMILES, UV dose, environment | ✅ Yes, for reproducibility |
| **Model output** | % conversion | ✅ Yes, for auditing |
| **Metadata** | Timestamp, model version | ✅ Yes, for versioning |

We do **not** save, on the other hand: intermediate images, embeddings (too large, reproducible), network errors (transient).

### 5.2 `app/models.py`

```python
"""
Database schema for the UV-curing predictor web app.

Uses rx.Model (SQLModel + Pydantic) so the schema is declared as a plain Python class.
Alembic migrations are auto-generated by reflex db make-migrations.
"""

from __future__ import annotations
from datetime import datetime, timezone
import reflex as rx


class CuringLog(rx.Model, table=True):
    """
    One row per prediction performed via the web app.

    Stores the user's raw input (molecule names, SMILES, parameters) plus
    the model output, so that any prediction can be audited and
    reproduced later.
    """

    # ---- User input ----
    pi_name: str
    monomer_name: str
    pi_smiles: str
    monomer_smiles: str
    environment: str                # "Solvent" or "Aqueous"
    uv_dose: float                  # mJ/cm²
    logp: float
    pi_concentration: float         # %

    # ---- Model output ----
    predicted_conversion: float     # 0–100

    # ---- Metadata ----
    created_at: datetime = datetime.now(timezone.utc)
```

**Why save both `pi_name` and `pi_smiles`?** Because `pi_name` is what the user typed (useful for analytics: "which molecules do people search for?"), while `pi_smiles` is what the model actually used (useful for reproducibility: if the user typed "TPO", the client resolved it to a specific SMILES — which one?).

### 5.3 Migrations
```powershell
reflex db init
reflex db make-migrations
reflex db migrate
```
**What each command does:**
* `db init` — initializes Alembic (creates the `alembic/` folder). Once per project.
* `db make-migrations` — compares the models in `app/models.py` with the current DB schema and generates a migration file with the "diff". Every time you change a model.
* `db migrate` — applies the migrations to the SQLite database.

The DB will be created as `app.db` (SQLite) at the project root.

---

## 6. Step 4: Build the Reactive State

### 6.1 Theory: Separating Input, Process, Output
A well-designed state has three groups of variables that do not mix:

```python
class PredictorState(rx.State):
    # ---- INPUT (mutated by users) ----
    pi_name: str = "Benzophenone"
    monomer_name: str = "Acrylic acid"
    environment: str = "Solvent"
    uv_dose: float = 150.0
    logp: float = 2.5
    pi_concentration: float = 2.0

    # ---- PROCESS (mutated by code while it works) ----
    is_loading: bool = False
    error_message: str = ""

    # ---- OUTPUT (mutated by code when it finishes) ----
    predicted_conversion: float = 0.0
    has_result: bool = False
```

**The advantage:** In the UI you can write `rx.cond(PredictorState.has_result, ...)` to show the result only after the first prediction, instead of showing an ambiguous `0.0`.

### 6.2 `app/state.py`

```python
"""
Reactive state for the UV-curing predictor web app.

Orchestration only: this module calls the inference pipeline and persists results.
It contains no ML logic itself.
"""

from __future__ import annotations
import reflex as rx

from app.models import CuringLog
from inference.pipeline import predict
from phase1.fetch_molecules_PIs import PI_CLIENT
from phase1.fetch_molecules_monomers import MONO_CLIENT


class PredictorState(rx.State):
    # ---- INPUT ----
    pi_name: str = "Benzophenone"
    monomer_name: str = "Acrylic acid"
    environment: str = "Solvent"
    uv_dose: float = 150.0
    logp: float = 2.5
    pi_concentration: float = 2.0

    # ---- PROCESS ----
    is_loading: bool = False
    error_message: str = ""

    # ---- OUTPUT ----
    predicted_conversion: float = 0.0
    has_result: bool = False

    # ---- RESOLVED SMILES (useful for display + audit) ----
    pi_smiles: str = ""
    monomer_smiles: str = ""

    @rx.event
    async def handle_prediction(self, form_data: dict):
        """
        Main event handler: resolve names, run the pipeline, save the log.

        Uses yield to push intermediate UI updates (spinner) before the
        heavy computation, so the user sees feedback immediately.
        """
        # ---- 1. Reset error state ----
        self.error_message = ""
        self.is_loading = True
        yield  # spinner appears in the browser

        # ---- 2. Read form ----
        try:
            pi_name = form_data.get("pi_name", "").strip()
            monomer_name = form_data.get("monomer_name", "").strip()
            environment = form_data.get("environment", "Solvent")
            uv_dose = float(form_data.get("uv_dose", 150.0))
            logp = float(form_data.get("logp", 2.5))
            pi_conc = float(form_data.get("pi_concentration", 2.0))

            if not pi_name or not monomer_name:
                raise ValueError("Both molecule names are required.")

            self.pi_name = pi_name
            self.monomer_name = monomer_name
            self.environment = environment
            self.uv_dose = uv_dose
            self.logp = logp
            self.pi_concentration = pi_conc

        except ValueError as exc:
            self.error_message = f"Invalid input: {exc}"
            self.is_loading = False
            yield
            return

        # ---- 3. Resolve names -> SMILES ----
        pi_entry = PI_CLIENT.fetch_single_molecule(pi_name, role="PI_TypeI")
        mono_entry = MONO_CLIENT.fetch_single_molecule(monomer_name, role="monomer")

        if pi_entry is None:
            self.error_message = f"Photoinitiator not found: {pi_name!r}"
            self.is_loading = False
            yield
            return
        if mono_entry is None:
            self.error_message = f"Monomer not found: {monomer_name!r}"
            self.is_loading = False
            yield
            return

        self.pi_smiles = pi_entry["smiles"]
        self.monomer_smiles = mono_entry["smiles"]

        # ---- 4. Run the inference pipeline ----
        try:
            conversion = predict(
                pi_smiles=self.pi_smiles,
                monomer_smiles=self.monomer_smiles,
                is_aqueous=1 if environment == "Aqueous" else 0,
                logp=logp,
                pi_concentration=pi_conc,
                uv_dose=uv_dose,
            )
        except Exception as exc:
            self.error_message = f"Prediction failed: {exc}"
            self.is_loading = False
            yield
            return

        self.predicted_conversion = round(conversion, 2)
        self.has_result = True

        # ---- 5. Persist to database ----
        try:
            with rx.session() as session:
                session.add(CuringLog(
                    pi_name=self.pi_name,
                    monomer_name=self.monomer_name,
                    pi_smiles=self.pi_smiles,
                    monomer_smiles=self.monomer_smiles,
                    environment=self.environment,
                    uv_dose=self.uv_dose,
                    logp=self.logp,
                    pi_concentration=self.pi_concentration,
                    predicted_conversion=self.predicted_conversion,
                ))
                session.commit()
        except Exception as exc:
            # Don't fail the prediction if the DB write fails
            print(f"[DB ERROR] {exc}")

        self.is_loading = False
        yield
```

**Critical teaching points:**
* **Strategic yields:** One before loading (shows spinner), one at the end (hides spinner). If an intermediate step fails, there is still a yield that resets `is_loading = False` and shows the error.
* **Layered error handling:** Input validation -> name not found -> pipeline error -> DB error. Each has a specific message for the user.
* **The DB is not critical:** If `session.commit()` fails, we log it but do not stop the response to the user. The result is already computed; losing it is less severe than crashing the app.
* **`role="PI_TypeI"` hardcoded:** This is the point where you decide the default. In the future you could add a dropdown "PI type" and pass the chosen role. For the MVP, `"PI_TypeI"` is a reasonable choice (it is the most widespread type).

---

## 7. Step 5: Build the UI

### 7.1 Theory: The UI as a Function of State
In Reflex, the UI contains no logic. It is a pure function that maps `state -> visual components`:

```python
# The UI "reads" the state and draws something
rx.text(PredictorState.predicted_conversion)   # <- reactive binding
```
When `predicted_conversion` changes on the server, Reflex knows that that specific `rx.text` depends on that variable, and updates only that DOM node. It does not reload the page.

### 7.2 `app/pages/index.py`

```python
"""
Main page of the UV-curing predictor.

The layout is deliberately split into three sections:
1. Input form (molecule names + environmental parameters)
2. Error banner (visible only if error_message is non-empty)
3. Result card (visible only if has_result is True)
"""

from __future__ import annotations
import reflex as rx
from app.state import PredictorState


def _input_section() -> rx.Component:
    """The form: two names + four parameters."""
    return rx.form(
        rx.vstack(
            rx.heading("Input parameters", size="4", margin_bottom="2"),

            rx.text("Photoinitiator name", font_weight="bold"),
            rx.input(
                name="pi_name",
                default_value=PredictorState.pi_name,
                placeholder="e.g. Benzophenone",
                width="100%",
            ),

            rx.text("Monomer name", font_weight="bold"),
            rx.input(
                name="monomer_name",
                default_value=PredictorState.monomer_name,
                placeholder="e.g. Acrylic acid",
                width="100%",
            ),

            rx.hstack(
                rx.box(
                    rx.text("Environment", font_weight="bold"),
                    rx.select(
                        ["Solvent", "Aqueous"],
                        name="environment",
                        default_value=PredictorState.environment,
                        width="100%",
                    ),
                    width="50%",
                ),
                rx.box(
                    rx.text("UV dose (mJ/cm²)", font_weight="bold"),
                    rx.input(
                        name="uv_dose",
                        type="number",
                        default_value=PredictorState.uv_dose.to_string(),
                        width="100%",
                    ),
                    width="50%",
                ),
                width="100%",
                spacing="4",
            ),

            rx.hstack(
                rx.box(
                    rx.text("LogP", font_weight="bold"),
                    rx.input(
                        name="logp",
                        type="number",
                        default_value=PredictorState.logp.to_string(),
                        width="100%",
                    ),
                    width="50%",
                ),
                rx.box(
                    rx.text("PI concentration (%)", font_weight="bold"),
                    rx.input(
                        name="pi_concentration",
                        type="number",
                        default_value=PredictorState.pi_concentration.to_string(),
                        width="100%",
                    ),
                    width="50%",
                ),
                width="100%",
                spacing="4",
            ),

            rx.button(
                rx.cond(
                    PredictorState.is_loading,
                    rx.hstack(rx.spinner(size="2"), rx.text("Computing..."), spacing="2"),
                    rx.text("Calculate curing conversion ⚡"),
                ),
                type="submit",
                color_scheme="sky",
                width="100%",
                margin_top="2",
                disabled=PredictorState.is_loading,
            ),
            spacing="3",
            width="100%",
        ),
        on_submit=PredictorState.handle_prediction,
        width="100%",
    )


def _error_banner() -> rx.Component:
    """Visible only when error_message is non-empty."""
    return rx.cond(
        PredictorState.error_message != "",
        rx.box(
            rx.hstack(
                rx.text("⚠️", font_size="1.5em"),
                rx.text(PredictorState.error_message, color="crimson"),
                spacing="3",
                align="center",
            ),
            border="1px solid crimson",
            border_radius="md",
            padding="3",
            background_color="#fff5f5",
            width="100%",
        ),
    )


def _result_card() -> rx.Component:
    """Visible only after the first successful prediction."""
    return rx.cond(
        PredictorState.has_result,
        rx.box(
            rx.vstack(
                rx.heading("Prediction result 📊", size="5"),
                rx.hstack(
                    rx.text("Double-bond conversion:", font_weight="semibold"),
                    rx.text(
                        f"{PredictorState.predicted_conversion}%",
                        font_weight="bold",
                        color=rx.cond(
                            PredictorState.predicted_conversion > 75,
                            "green",
                            "orange",
                        ),
                    ),
                    spacing="3",
                ),
                rx.text(
                    f"PI: {PredictorState.pi_name}  |  "
                    f"Monomer: {PredictorState.monomer_name}",
                    color_scheme="gray",
                    font_size="0.9em",
                ),
                spacing="2",
                align="start",
            ),
            border="2px solid",
            border_color=rx.cond(
                PredictorState.predicted_conversion > 75,
                "green",
                "orange",
            ),
            border_radius="lg",
            padding="5",
            background_color=rx.cond(
                PredictorState.predicted_conversion > 75,
                "#f0fff4",
                "#fffaf0",
            ),
            width="100%",
        ),
    )


def index() -> rx.Component:
    """The home page."""
    return rx.container(
        rx.vstack(
            rx.heading("AI-Powered UV-Curing Predictor 🧪", size="8", margin_y="4"),
            rx.text(
                "Enter a photoinitiator and a monomer by name. "
                "The SMILES are resolved automatically via PubChem, "
                "then the model predicts the theoretical double-bond conversion.",
                color_scheme="gray",
                text_align="center",
                max_width="600px",
            ),

            rx.box(
                rx.vstack(
                    _input_section(),
                    _error_banner(),
                    spacing="4",
                    width="100%",
                ),
                width="600px",
                border="1px solid #e2e8f0",
                border_radius="lg",
                padding="6",
                background_color="white",
            ),

            _result_card(),

            align="center",
            spacing="6",
            padding_y="10",
            width="100%",
        ),
        size="3",
    )
```

**Recurring UI patterns:**
* `rx.cond(condition, component_if_true, component_if_false)`: reactive conditional rendering.
* `PredictorState.var.to_string()`: some properties (like input `default_value`) require strings, not numbers. `.to_string()` is the method that does the conversion.
* **Composition into functions:** `_input_section`, `_error_banner`, `_result_card`: each section of the UI is a function. This makes the code readable and allows each piece to be visually tested in isolation.

### 7.3 `app/app.py` (Entry Point)

```python
"""
Reflex app entry point.

Run with: reflex run
The app serves the home page at http://localhost:3000 and the backend API at http://localhost:8000.
"""

import reflex as rx
from app.pages.index import index

app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="sky",
        radius="medium",
    ),
)
app.add_page(index, route="/", title="UV-Curing Predictor")
```

---

## 8. Testing Strategy

### 8.1 The Three Levels of Testing

| Level | What it tests | How |
| :--- | :--- | :--- |
| **Unit** | `inference/pipeline.predict` | Standalone Python script |
| **Integration** | PubChem client inside the state | Script that instantiates `PredictorState` |
| **End-to-end** | Full UI | `reflex run` + manual click |

### 8.2 Unit Test (Already Seen in 3.4)

```python
# inference/test_pipeline.py
from inference.pipeline import predict

def test_known_molecules():
    result = predict(
        pi_smiles="O=C(C1=CC=CC=C1)C2=CC=CC=C2",
        monomer_smiles="C=CC(=O)O",
        is_aqueous=0,
        logp=3.0,
        pi_concentration=2.0,
        uv_dose=150.0,
    )
    assert 0.0 <= result <= 100.0, result
    print(f"OK: {result:.2f}%")

if __name__ == "__main__":
    test_known_molecules()
```

### 8.3 Integration Test (State Without UI)
Reflex allows testing the state without starting the server, but that is an advanced topic. For the MVP it is enough to verify that importing the app does not fail:

```powershell
python -c "from app.state import PredictorState; print('OK')"
```
If this command completes without errors and without having called PubChem or TensorFlow, you have verified that lazy loading works.

### 8.4 End-to-End Test
Launch the app and try:
* A pair of known molecules ("Benzophenone" + "Acrylic acid")
* A non-existent name ("Fake_aspirin") -> should show error banner, not crash
* A mixed-case name ("BENZOPHENONE") -> should work (case-insensitive)

---

## 9. Running the App

```powershell
reflex run
```
Reflex prints something like:
```text
INFO  Starting Reflex in development mode
INFO  Frontend running at http://localhost:3000
INFO  Backend running at http://localhost:8000
```

Open `http://localhost:3000` in the browser. The first prediction will take 4–6 seconds (TensorFlow loading); all subsequent ones ~200 ms. If you see a spinner while it computes, `yield` works. If you see a blank page, check the terminal console for Python errors.

The `app.db` database is created automatically at the root on the first write.

---

## 10. Summary and Next Steps

### 10.1 What You Have Built
```text
+-------------------------------------------------------------+
|                    Phase 5 Architecture                     |
+-------------------------------------------------------------+
|  [Browser]                                                  |
|      |  WebSocket                                           |
|      v                                                      |
|  [Reflex State]                                             |
|      |                                                      |
|      |-- PI_CLIENT / MONO_CLIENT   (name -> SMILES)          |
|      |-- inference.pipeline.predict (SMILES -> % conversion) |
|      |       |                                              |
|      |       |-- RDKit -> grayscale image                    |
|      |       |-- MobileNetV2 -> 1280-D embedding (lazy)      |
|      |       +-- XGBoost -> prediction (lazy)                |
|      |                                                      |
|      +-- CuringLog (SQLModel) -> SQLite                      |
+-------------------------------------------------------------+
```

### 10.2 The Key Architectural Decisions
1. **Separation of UI / logic / ML:** Three folders, three responsibilities.
2. **Lazy singleton for heavy models:** First click slow, all others fast.
3. **`yield` in every async handler:** Immediate feedback to the user.
4. **Layered errors:** Input, lookup, pipeline, DB — each with a specific message.
5. **The DB does not block the response:** If the write fails, the user still sees the result.
6. **PI role hardcoded for the MVP:** A postponable choice, not a binding one.

### 10.3 What to Postpone to Post-MVP

| Feature | Why postpone it |
| :--- | :--- |
| **User auth (`reflex-local-auth`)** | Not essential to demonstrate the pipeline |
| **Prediction history visible in UI** | The DB already saves them, just needs a page |
| **PI role selection (Type I/II)** | A dropdown, but requires training labels |
| **Embedding cache per SMILES** | RDKit + MobileNetV2 at 200 ms are acceptable for the MVP |
| **Deployment (Vercel, Fly.io)** | Local `reflex run` is enough for the demo |

### 10.4 The Final README
At the root, a `README.md` explaining:
* What the project does (predict UV conversion)
* How to install (`pip install -r requirements.txt`)
* How to prepare the data (`python -m phase1.fetch_molecules_PIs` etc.)
* How to train the models (`python phase4/train_xgboost.py`)
* How to launch the app (`reflex run`)
* How to test (`python inference/test_pipeline.py`)

This closes the MVP loop: anyone who clones the repo can reproduce the entire pipeline from scratch.
