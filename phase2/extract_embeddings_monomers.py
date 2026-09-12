import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D


# ==================== CONFIGURATION ====================

# folder in which this script is located (phase2/)
BASE_DIR = Path(__file__).resolve().parent
# data dir path in this sub-folder
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
# folder in which NPZ file we need to access to is located (in phase1/)
IMAGES_P1_DIR = BASE_DIR.parent / "phase1" / "images"
# folder in which metadata file we need to access to is located (in phase1/)
DATA_P1_DIR = BASE_DIR.parent / "phase1" / "data"

# inputs
# NPZ file path from phase 1 (monomers)
INPUT_NPZ = IMAGES_P1_DIR / "molecular_images_monomers.npz"
# metadata file path from phase 1 (monomers)
INPUT_META = DATA_P1_DIR / "molecular_metadata_monomers.csv"

# outputs
OUTPUT_EMBED = DATA_DIR / "embeddings_monomers.npy"
OUTPUT_META = DATA_DIR / "embeddings_metadata_monomers.csv"


# ==================== 1. IMAGES LOADING ====================

data = np.load(INPUT_NPZ)
images = data['images'] 
meta = pd.read_csv(INPUT_META)

print(f'{len(images)} images loaded.')
print(f'    Shape of an image: {images[0].shape}')
print(f'    Metadata: {len(meta)} rows')
print(f'    Columns: {meta.columns.tolist()}')


# ==================== 2. IMAGES PREPROCESSING ====================

'''
Pre‑trained models expect input in a specific format. 

For MobileNetV2 in Keras:
    - Dimensions: 224 × 224 pixels.
    - Channels: 3 (RGB). We need to convert grayscale (1 channel) 
      to RGB by repeating the same channel 3 times.
    - Data type: float32.
    - Pixel range: 0‑1 (normalisation) – or use the model’s own 
      preprocessing function.
    - Batch dimension: add an extra axis for batch processing.'''

print("Images preprocessing started...")

# 1. from grey scale to RGB
# (224, 224, 1) → (224, 224, 3)
# we repeat a channel 3 times: no new infos added, 
# just adaptation to MobileNetV2 contraints
images_rgb = np.stack([images] * 3, axis=-1)
print(f'  RGB shape: {images_rgb.shape}')

# 2. pixels normalization for MobileNetV2
# converto to float32 and apply preprocessing
images_pp = preprocess_input(images_rgb.astype(np.float32))
print(f'  Preprocessed shape: {images_pp.shape}')
print(f'  Min value: {images_pp.min():.2f}, Max value: {images_pp.max():.2f}')


# ==================== 3. LOAD MODEL ====================

print("Loading of MobileNetV2...")

# 1. loading pre trained model
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# 2. freeze all the layers (won't be trained)
'''
Normally, weights are updated at each batch to reduce error.

Withy freezing, weights never change: they are blocked at the 
same value they had at the loading moment.
  - First layers already know how to recognize universal patterns.
  - Amount of data available must be considered: training of many layers
    on a short range of data gives overfitting.
Only final layers (the ones we are going to add) are going to be trained,
because they will be better able to well recognize images without generalizing
and also they have a lot parameters less compared to the first ones (less overfitting).
'''
base_model.trainable = False

print(f'Model successfully loaded and frozen.')
print(f'  Total layers: {len(base_model.layers)}')
print(f'  Trainable layers: {sum(1 for layer in base_model.layers if layer.trainable)}')

# uncomment the line below to see model architecture in the output
# base_model.summary()

# 3. application of Global Average Pooling to the model output and embedding model creation
# if not done, it would return the output of the last convolutional layer (out_relu). 
# This output is still spatial (7×7); it has not yet been compressed into a vector.
# we take care of this manually

# create Global Average Pooling layer
gap_layer = GlobalAveragePooling2D()

# apply pooling to model output
embeddings_output = gap_layer(base_model.output)

# embedding model creation to extract output from Global Average Pooling
embedding_model = Model(
    inputs=base_model.input,
    outputs=embeddings_output
)

print("Embedding model successfully created.")
print(f'  Output shape: {embedding_model.output_shape}') # (None, 1280)


# ==================== 4. EMBEDDINGS EXTRACTION ====================
'''
The model is now ready.
We extract all vectors for the monomer images.
'''

print("Extracting embeddings...")
embeddings = embedding_model.predict(images_pp, batch_size=32, verbose=1)

print(f'Embeddings successfully extracted.')
print(f'  Shape: {embeddings.shape}') # (40, 1280)


# ==================== 5. SAVE ====================

np.save(OUTPUT_EMBED, embeddings)
meta.to_csv(OUTPUT_META, index=False)

print(f'\nEmbeddings saved to {OUTPUT_EMBED}')
print(f'Metadata saved to {OUTPUT_META}')


'''
==================== OUTPUT ====================
40 images loaded.
    Shape of an image: (224, 224)
    Metadata: 40 rows
    Columns: ['name', 'smiles', 'role', 'augment']
Images preprocessing started...
  RGB shape: (40, 224, 224, 3)
  Preprocessed shape: (40, 224, 224, 3)
  Min value: -1.00, Max value: 1.00
Loading of MobileNetV2...
2026-09-05 20:34:40.256172: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: SSE3 SSE4.1 SSE4.2 AVX AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
Model successfully loaded and frozen.
  Total layers: 154
  Trainable layers: 0
Embedding model successfully created.
  Output shape: (None, 1280)
Extracting embeddings...
2/2 ━━━━━━━━━━━━━━━━━━━━ 4s 1s/step
Embeddings successfully extracted.
  Shape: (40, 1280)

Embeddings saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase2\data\embeddings_monomers.npy
Metadata saved to D:\Irene\Desktop\AI_&_Data_Science_training_BeCode\BeCode_Projects\specialization\AI-Powered-UV-Curing-Predictor-and-PI-Discovery\phase2\data\embeddings_metadata_monomers.csv
'''