'''
Reactive state for the UV-curing predictor web app.

Orchestration only: this module calls the inference pipeline and persists
results. It contains no ML logic itself.
'''

from __future__ import annotations

import reflex as rx

from app.models import CuringLog
from inference.pipeline import predict
from phase1.fetch_molecules_PIs import PI_CLIENT
from phase1.fetch_molecules_monomers import MONO_CLIENT


class PredictorState(rx.State):
    # ---- INPUT (mutated by the user via the form) ----
    pi_name: str = "Benzophenone"
    monomer_name: str = "Acrylic acid"
    environment: str = "Solvent"
    uv_dose: float = 150.0
    logp: float = 2.5
    pi_concentration: float = 2.0

    # ---- PROCESS (mutated by the code while it works) ----
    is_loading: bool = False
    error_message: str = ""

    # ---- OUTPUT (mutated by the code when it finishes) ----
    predicted_conversion: float = 0.0
    has_result: bool = False

    # ---- RESOLVED SMILES (for display + audit) ----
    pi_smiles: str = ""
    monomer_smiles: str = ""

    @rx.event
    async def handle_prediction(self, form_data: dict):
        """
        Main event handler: resolve names -> SMILES, run the pipeline,
        save the log. Uses yield to push intermediate UI updates
        (spinner) before the heavy computation.
        """
        # ---- 1. Reset error state, show spinner ----
        self.error_message = ""
        self.is_loading = True
        yield

        # ---- 2. Read and validate form ----
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

        except (ValueError, TypeError) as exc:
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

        # ---- 5. Persist to database (non-critical) ----
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
            print(f"[DB ERROR] {exc}")

        self.is_loading = False
        yield