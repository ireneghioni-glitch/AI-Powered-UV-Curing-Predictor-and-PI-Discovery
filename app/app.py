"""
Reflex app entry point.

Run with:
    reflex run

The app serves the home page at http://localhost:3000
and the backend API at http://localhost:8000.
"""

import reflex as rx

import app.models  # required for Alembic to discover CuringLog
from app.pages.index import index


app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="sky",
        radius="medium",
    ),
)
app.add_page(index, route="/", title="UV-Curing Predictor")