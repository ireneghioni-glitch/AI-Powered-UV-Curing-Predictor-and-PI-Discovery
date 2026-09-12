# Migration from rx.Model to Pure SQLModel — Post-MVP Guide

> **Important Note:** To be executed only after the application works end-to-end and the MVP is demonstrable. Do not execute now. The deprecation warning is cosmetic and does not block execution.

---

## 0. Context: What Changes and Why

| Aspect | `rx.Model` (Current) | Pure `SQLModel` (Target) |
|---|---|---|
| **Base Class** | `rx.Model` (Reflex wrapper) | `SQLModel` (Library) |
| **`id` Field** | Added automatically | Must be declared explicitly |
| **Session** | `rx.session()` | `Session(engine)` or `rx.session()` (verify) |
| **Reflex Dependency** | Yes | No |
| **Deprecation Status** | Yes, removed in Reflex 1.0 | No, standard future-proof approach |

**Reason for migration:** Prepare for Reflex 1.0 by avoiding sudden breaking changes. It is not urgent as long as you stay on version 0.9.x.

---

## 1. Pre-flight: Check Current State

Before touching anything, capture the current state of your setup and database.

```powershell
### 1. Database Backup
Copy-Item app.db app.db.bak

### 2. Backup files that will be modified
Copy-Item app\models.py app\models.py.bak
Copy-Item app\state.py app\state.py.bak

### 3. Check current database schema
@'
import sqlite3
conn = sqlite3.connect("app.db")
cur = conn.cursor()
cur.execute("SELECT sql FROM sqlite_master WHERE name='curinglog'")
print(cur.fetchone()[0])
conn.close()
'@ | Out-File -Encoding utf8 check_schema.py
python check_schema.py
Remove-Item check_schema.py
```

*Save the output of `check_schema.py`: this is the current table schema. You will need it to compare after the migration.*

---

## 2. Verify Requirements

Check what `rx.Model` provides that you must replicate manually:

```powershell
python -c "import reflex as rx; print(rx.Model.__mro__)"
```
**Expected Output:** A list that includes `SQLModel` and `SQLModelMetaclass` somewhere. This confirms that `rx.Model` is merely a thin wrapper around `SQLModel`.

Check for the presence of an auto-generated `id` field:

```powershell
python -c "from app.models import CuringLog; print(CuringLog.__table__.columns.keys())"
```
**Expected Output:** Something like `['id', 'pi_name', 'monomer_name', ...]`. Note the `id` field that you did not declare explicitly: `rx.Model` added it automatically for you.

---

## 3. Modify `app/models.py`

### 3.1 — Change Base Class and Add Explicit `id`

**Before (Current):**
```python
class CuringLog(rx.Model, table=True):
    # ... fields ...
```

**After:**
```python
from sqlmodel import SQLModel, Field
from datetime import datetime, timezone
from typing import Optional

class CuringLog(SQLModel, table=True):
    # Explicit id (rx.Model used to add this automatically)
    id: Optional[int] = Field(default=None, primary_key=True)

    # ... remaining fields unchanged ...
    pi_name: str
    monomer_name: str
    # ...
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
```

**Key Changes:**
* **Base class:** `rx.Model` → `SQLModel`
* **Imports:** Change from `reflex` to `sqlmodel`
* **Added `id`:** `Optional[int] = Field(default=None, primary_key=True)` — mandatory in pure SQLModel
* **Cleanup:** Removed `import reflex as rx` if not used elsewhere in the file

### 3.2 — Verify Importability

```powershell
python -c "from app.models import CuringLog; print(CuringLog.__table__.columns.keys())"
```
**Expected Output:** The exact same list as before, with `id` as the first element.

---

## 4. Modify `app/state.py`

The current code uses `with rx.session() as session:`. In theory, it should still work (`rx.session()` is just a wrapper around a SQLModel session), but the deprecation message suggests using SQLModel directly. You have two options:

### Option A — Keep `rx.session()` (Simpler)
Do not change anything in `state.py`. Try running the app: if it works, you are done.

### Option B — Use SQLModel Directly (Cleaner, Future-proof)
```python
### app/state.py
from sqlmodel import Session
from reflex import get_engine  # or from the Reflex configuration

### ... inside the handle_prediction method ...
with Session(get_engine()) as session:
    session.add(CuringLog(...))
    session.commit()
```

*Note:* The exact way to obtain the engine from Reflex varies by version. Consult the official documentation for your specific version:

```powershell
python -c "import reflex as rx; print([n for n in dir(rx) if 'engine' in n.lower() or 'session' in n.lower()])"
```
You will see the available names. On Reflex 0.9.x, `rx.session()` remains supported as a shortcut.

**Recommendation:** Try Option A first. If it works, the migration is complete with only one modified file. If `rx.session()` is removed in Reflex 1.0, switch to Option B.

---

## 5. Handling Alembic Migrations

This is the delicate part. The `curinglog` table already exists in the DB, created by `rx.Model`. Changing the base class does not change the schema (the fields are identical; `id` exists in both cases), so Alembic should not detect any differences.

### 5.1 — Verify No Schema Differences

```powershell
reflex db makemigrations --message "post rx.Model migration check"
```

**Outcomes:**
* `"No changes detected"` (or similar) → ✅ Migration is zero-cost. No DB changes required.
* **Generates a migration file** → ⚠️ Something changed. Open the file to see what:
  ```powershell
  Get-ChildItem alembic\versions | Sort-Object LastWriteTime | Select-Object -Last 1
  ```
  * If the file contains only `pass` inside `upgrade()` and `downgrade()`, you can delete it:
    ```powershell
    Remove-Item alembic\versions\<empty-file>.py
    ```
  * If it contains `op.alter_column` or similar, evaluate whether to apply it. Usually, these are cosmetic differences (e.g., `VARCHAR` vs `TEXT`) that SQLite ignores.

### 5.2 — Restarting from Scratch (Nuclear Option)

If something is confusing and the DB does not contain critical data:

```powershell
### Backup
Copy-Item app.db app.db.old

### Full Reset
Remove-Item app.db
Remove-Item -Recurse alembic\versions\*.py

### Regenerate from scratch
reflex db makemigrations --message "initial schema with SQLModel"
reflex db migrate
```
*Warning:* This deletes all existing data. Only do this if the DB is empty or for testing purposes.

---

## 6. Testing the Migration

### 6.1 — Import Safety
```powershell
python -c "from app.state import PredictorState; from app.models import CuringLog; print('OK')"
```

### 6.2 — End-to-End Test
```powershell
reflex run
```
Open `http://localhost:3000`, run a prediction (e.g., Benzophenone + Acrylic acid), and verify that:
1. The result appears correctly in the UI.
2. No `DeprecationWarning: reflex.Model...` appears in the terminal.
3. No session errors occur.

### 6.3 — Verify Log Written to DB
```powershell
@'
import sqlite3
conn = sqlite3.connect("app.db")
cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM curinglog")
print(f"Rows in curinglog: {cur.fetchone()[0]}")
cur.execute("SELECT pi_name, monomer_name, predicted_conversion FROM curinglog LIMIT 3")
for row in cur.fetchall():
    print(row)
conn.close()
'@ | Out-File -Encoding utf8 check_rows.py
python check_rows.py
Remove-Item check_rows.py
```
**Expected Output:** At least one row containing the prediction data just executed.

---

## 7. Rollback (If Something Goes Wrong)

```powershell
### Restore files
Copy-Item app\models.py.bak app\models.py
Copy-Item app\state.py.bak app\state.py

### Restore DB
Remove-Item app.db
Copy-Item app.db.bak app.db

### Verify Rollback
python -c "from app.state import PredictorState; print('rollback OK')"
```

---

## 8. Final Cleanup

Once everything is verified and working:

```powershell
Remove-Item app\models.py.bak
Remove-Item app\state.py.bak
Remove-Item app.db.bak
```

---

## 9. Summary Checklist

- [ ] Backup `app.db`, `models.py`, `state.py`
- [ ] Save current schema with `check_schema.py`
- [ ] Modify `models.py`: `rx.Model` → `SQLModel` + add explicit `id`
- [ ] Verify importability of `CuringLog`
- [ ] Decide whether to modify `state.py` (Option A or B)
- [ ] Run `reflex db makemigrations` → check if differences are detected
- [ ] Run `reflex db migrate` → apply changes if needed
- [ ] Test: import, `reflex run`, verify rows in DB
- [ ] If OK: remove backups
- [ ] If broken: rollback using `.bak` files

---

## 10. When to Do This

Do **not** do this right now. Do it when:
* The MVP is demonstrated end-to-end (browser → prediction → DB)
* The repository is committed and stable
* You have 30–60 minutes to perform the migration calmly and test
* Ideally before updating Reflex to version 1.0, so you avoid broken code

Log this as technical debt in `README.md` or a `TODO.md` file:
```text
* [ ] Migrate app/models.py from rx.Model to SQLModel (deprecated in Reflex 0.9.2, removed in 1.0)
      See migration guide in docs/ or conversation log.
```
This way, when you return to the project later, you will know exactly what to do and where to find the procedure.
