'''
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
'''

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

from shared.molecule_images import smiles_to_grayscale, IMG_SIZE


# ==================== PATHS ====================
BASE_DIR = Path(__file__).resolve().parent.parent
PHASE4 = BASE_DIR / "phase4"
XGBOOST_MODEL_PATH = PHASE4 / "xgboost_model.json"


# ==================== LAZY SINGLETONS ====================
# The cost is paid only once, on the first prediction. 
# Subsequent ones are instantaneous (milliseconds).
_mobilenet_model = None    # set by _get_mobilenet()
_xgboost_model = None      # set by _get_xgboost()

def _get_mobilenet():
    '''
    Lazily load the frozen MobileNetV2 embedding extractor.

    The first call imports TensorFlow (~3-4 s) and builds the model.
    Subsequent calls return the cached instance in O(1).
    '''
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
    '''
    Lazily load the trained XGBoost model from disk.
    '''
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

def image_to_embedding(image: np.ndarray) -> np.ndarray:
    '''
    Pass a grayscale image through frozen MobileNetV2.

    Returns a 1-D array of shape (1280,).

    NOTE
    ----
    The equivalent batch code lives in phase2/extract_embeddings.py, but
    that script is not import-safe (it runs the full batch at import time).
    Rather than refactor Phase 2 mid-MVP, we re-implement the two lines
    needed at runtime. If a future refactor extracts Phase 2 into a
    callable module, this function should be replaced by an import, in the
    same way smiles_to_grayscale was.
    '''
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    # Grayscale -> RGB by channel replication (matches Phase 2 preprocessing)
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
    '''
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
        Photoinitiator concentration in percent (typically 1-5).
    uv_dose : float
        UV energy dose in mJ/cm2 (typically 50-500).

    Returns
    -------
    float
        Predicted double-bond conversion in percent, clipped to [0, 100].
    '''
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
