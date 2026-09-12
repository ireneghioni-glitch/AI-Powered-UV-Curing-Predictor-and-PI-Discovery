# Reflex — Essential Workflow Sequence

## A. One-Time Setup (Per Machine)
- **Create dedicated conda environment:** `conda create -n reflex_app python=3.11 -y`
- **Activate it:** `conda activate reflex_app`
- **Install Reflex:** `pip install reflex`
- **Install Node.js inside environment:** `conda install -c conda-forge "nodejs>=22.22" -y`
- **Verify:** `reflex --version` and `node --version`

---

## B. One-Time Setup (Per Project)
- **From project root:** `reflex init`
- **Select template:** Blank (`0`)
- **Rename generated folder to:** `app/`
- **Rename internal file to:** `app/app.py`
- **Update `rxconfig.py`:** `app_name="app"`
- **Test:** `reflex run` → open `http://localhost:3000` → verify *"Welcome to Reflex!"*

---

## C. Final App Structure

```text
app/
├── app.py            ← entry point (rx.App, add_page)
├── models.py         ← DB tables (rx.Model)
├── state.py          ← reactive logic (rx.State)
└── pages/
    ├── __init__.py
    └── index.py      ← page UI
```

---

## D. Development Cycle (Repeat for each feature)
1. **Write DB model** in `app/models.py`
2. **Apply migration:** `reflex db init` (once) → `reflex db make-migrations` → `reflex db migrate`
3. **Write state** in `app/state.py` (variables + event handlers)
4. **Write UI** in `app/pages/index.py` (`rx.*` components)
5. **Register page** in `app/app.py` with `app.add_page(...)`
6. **Launch:** `reflex run` → test in browser → `Ctrl+C` to stop

---

## E. Mental Model of the Reactive Cycle

```text
[Browser] --click--> [Event Handler in State]
                         │
                         ├─ mutates state variables
                         ▼
             [Reflex re-renders]
                         ▼
[Only the UI part dependent on changed variables updates]
```

---

## F. Golden Rules
- **State vars:** Everything the UI reads must be declared at the class level, otherwise Reflex will not react [816].
- **`yield` in async handlers:** Acts as a checkpoint sending the state to the browser (used for spinners/loading indicators) [816, 817].
- **Heavy logic outside state (`inference/`, `shared/`):** State orchestrates, it does not compute [817].
- **Heavy ML models:** Load them using lazy singletons, never at import-time [817].
- **`reflex run`:** Frontend on `:3000`, backend on `:8000`, hot-reload active [817].
- **DB commands:** Always execute from the project root with the `reflex_app` environment active [817].

---

## G. Next Concrete Step
We are at point C, ready for step 12: writing `app/models.py` [817].
