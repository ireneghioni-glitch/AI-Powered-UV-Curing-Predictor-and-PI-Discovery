"""
Throwaway end-to-end check for inference/pipeline.py.

Verifies that the full runtime chain works:
    SMILES -> grayscale image -> MobileNetV2 embedding -> XGBoost prediction

Also verifies the lazy-singleton pattern: the FIRST call pays the model
loading cost (~5-8 s), subsequent calls in the SAME PROCESS are fast
(~200 ms). Note: this benefit does NOT carry across separate Python
invocations, because each `python -m ...` starts a fresh process and
re-initializes all module globals.

Run from the PROJECT ROOT:

    python -m phase1.exploratory.test_pipeline
"""

import time

from inference.pipeline import predict


def _one_prediction(label: str) -> float:
    t0 = time.perf_counter()
    result = predict(
        pi_smiles="O=C(C1=CC=CC=C1)C2=CC=CC=C2",   # Benzophenone
        monomer_smiles="C=CC(=O)O",                 # Acrylic acid
        is_aqueous=0,
        logp=3.0,
        pi_concentration=2.0,
        uv_dose=150.0,
    )
    elapsed = time.perf_counter() - t0
    print(f"{label}: {result:.2f}%  ({elapsed:.3f} s)")
    return result


def main():
    r1 = _one_prediction("First call  (loads models)")
    r2 = _one_prediction("Second call (cached)     ")
    r3 = _one_prediction("Third call  (cached)     ")

    assert r1 == r2 == r3, "Predictions are not deterministic!"
    assert 0.0 <= r1 <= 100.0, f"Out of range: {r1}"

    print("\nAll three calls returned the same value and were in range.")


if __name__ == "__main__":
    main()