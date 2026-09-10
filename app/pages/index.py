'''
Main page of the UV-curing predictor.

The layout is split into three sections:
1. Input form (molecule names + environmental parameters)
2. Error banner (visible only if error_message is non-empty)
3. Result card (visible only after the first successful prediction)
'''

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
                    rx.hstack(
                        rx.spinner(size="2"),
                        rx.text("Computing..."),
                        spacing="2",
                    ),
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