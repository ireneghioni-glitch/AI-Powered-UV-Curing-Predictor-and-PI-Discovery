# Phase 2 Development Log: From Molecular Images to Embeddings

## Document Purpose

This document chronicles the complete development journey of the `extract_embeddings.py` script, the core component of Phase 2 of the MVP pipeline. It explains how we transformed 208 molecular images into dense vector representations (embeddings) using a pre-trained Convolutional Neural Network (MobileNetV2) via transfer learning.

This log covers:
- The theoretical foundations of CNNs and transfer learning
- Step-by-step implementation decisions
- The reasoning behind each architectural choice
- Obstacles encountered and how they were resolved
- Final results and deliverables

---

## 1. The Original Specification (What Phase 2 Required)

The MVP Technical Specification (curing-prediction-pipeline-mvp-v3-en.md) defines Phase 2 as:

> **"Computer Vision and Feature Extraction"**:
> 1. **Model Import**: Load a lightweight convolutional network (MobileNetV2 is ideal for fast computation on CPU).
> 2. **Layer Freezing**: Set `trainable = False` on all convolutional layers of the network.
> 3. **Visual Embeddings Generation**: Pass the grayscale molecular drawing through the network. Extract the output of the penultimate layer (Global Average Pooling). You will obtain a dense vector (e.g., of 1024 numbers) that mathematically describes the "geometry" of the molecule.

The specification outlines a clear workflow:

    [Images NPZ] 
        → Preprocess (grayscale → RGB, normalize) 
        → MobileNetV2 (frozen) 
        → Global Average Pooling 
        → Embeddings (1280 features per image)

This is exactly what extract_embeddings.py implements.

---

## 2. The Challenge: From Pixels to Features

At the end of Phase 1, we had:
- 208 images (52 molecules × 4 augmentations)
- Each image: 224×224 pixels, grayscale
- Format: NPZ (compressed NumPy array)

### The Problem with Raw Pixels

Raw pixels are:
- High-dimensional: 224×224 = 50,176 numbers per image
- Redundant: Adjacent pixels are highly correlated
- Lacking semantic meaning: The values don't tell us "this is an aromatic ring" or "this is a carbonyl group"

### The Solution: Embeddings

An embedding is a dense, lower-dimensional vector that captures the semantic content of an image. Instead of using 50,176 raw pixels, we compress each image into a vector of 1280 numbers that represents its visual "fingerprint".

Why this helps:
- Rich features (each number represents a learned pattern)
- Reduced dimension (99% compression)
- Better for regression (the vectors are more linearly separable)

Reference: 02-intro_to_computer_vision_pytorch.pdf – Section 1: "Images as Tensors"

---

## 3. Theoretical Foundations: CNNs and Transfer Learning

### 3.1 What is a Convolutional Neural Network (CNN)?

A CNN is a special type of neural network designed to process grid-like data (such as images). It learns to detect patterns by applying small filters (kernels) that slide over the image.

Reference: 03-the-deep-in-deep-learning.pdf – Section on Forward Pass

#### Key CNN Concepts

| Concept | Definition |
|---------|------------|
| Local Connectivity | Each neuron connects only to a small region of the input (receptive field) |
| Parameter Sharing | The same filter is applied across the entire image |
| Hierarchical Feature Learning | Early layers detect simple features (edges, corners); deeper layers combine them into complex structures (eyes, wheels, functional groups) |

Reference: 01-image-classification-theory.md – Section "CNN Image Classification"

#### Why CNNs Work So Well for Images

1. Local Connectivity: Images have strong spatial correlation. Pixels near each other are related (bonds, rings). You don't need to look at the entire molecule to detect a double bond in one area.

2. Parameter Sharing: A filter that detects a vertical edge in one location is the same filter needed to detect a vertical edge anywhere else. No need to re-learn for every position.

3. Hierarchical Feature Learning: 
   - Early layers: detect edges, corners, simple shapes
   - Middle layers: combine into functional groups (carbonyl, hydroxyl, aromatic rings)
   - Late layers: combine into whole molecular "fingerprints"

### 3.2 Transfer Learning

Training a CNN from scratch requires:
- Millions of labelled images (ImageNet has 14 million)
- Massive computational resources (GPUs for days or weeks)
- Expertise in hyperparameter tuning

Transfer learning solves this by taking a model already pre-trained on a large dataset and reusing its learned features for a new task.

Reference: 15-minute - Women TechMaker Brussel - Slide Deck.pdf – Slides 28-31

#### Why Transfer Learning Works

The early layers of a CNN learn general-purpose features (edges, textures, simple shapes) that are useful for almost any image recognition task. Only the later layers learn task-specific patterns.

My approach:
1. Load a pre-trained CNN (MobileNetV2) with ImageNet weights
2. Freeze all convolutional layers (prevent updates during training)
3. Use the frozen model as a fixed feature extractor
4. Extract the output of the Global Average Pooling layer (our embedding)

Reference: 04-object_detection_pytorch_example.pdf – Part 2: "Build the model: swap the head, keep the pretrained backbone"

---

## 4. Why MobileNetV2?

| Advantage | Reason |
|-----------|--------|
| Lightweight | Designed for mobile and embedded vision; runs fast on CPU |
| Pre-trained on ImageNet | Has learned to recognize edges, textures, and shapes – useful for molecular structures |
| Global Average Pooling output | Produces a dense vector of 1280 features per image – manageable for our regressor |
| Widely supported | Available in both PyTorch and Keras (tensorflow.keras.applications) |

MobileNetV2 was chosen over larger models (ResNet50, VGG16) because:
- It is lighter and faster (important for our MVP)
- It still provides high-quality features (1280 dimensional embedding)
- The size of our dataset (208 images) doesn't require the capacity of a larger model

Reference: curing-prediction-pipeline-mvp-v3-en.md – Phase 2: "Model Import"

---

## 5. The Development Journey (Step-by-Step)

### 5.1 Project Structure

Code has been organized by phase:

    AI-Powered-UV-Curing-Predictor-and-PI-Discovery/   (root)
    ├── phase1/                                        (Phase 1)
    │   ├── data/
    │   │   ├── molecules_PIs.csv
    │   │   └── molecular_metadata.csv
    │   └── images/
    │       └── molecular_images.npz
    └── phase2/                                        (Phase 2)
        ├── data/                                      (Phase 2 outputs)
        └── extract_embeddings.py                      (the script)

Why this structure:
- Separates phases logically
- Prevents confusion between scripts and data
- Scales well for future phases (3, 4, 5)

### 5.2 Step 1: Loading Images

    data = np.load(INPUT_NPZ)
    images = data['images']  # (208, 224, 224)
    meta = pd.read_csv(INPUT_META)

Things learned:
- NPZ is a compressed NumPy format – efficient and fast
- The images are grayscale (1 channel, 224×224)
- We need metadata (name, role, augment) to trace each embedding back to its source

### 5.3 Step 2: Preprocessing for MobileNetV2

#### 5.3.1 From Grayscale to RGB

    images_rgb = np.stack([images] * 3, axis=-1)

Why this step is needed: 
- MobileNetV2 expects 3-channel RGB images
- Our images are grayscale (1 channel)
- We repeat the same channel 3 times (no new info, just compatibility)

Reference: 00-preprocessing_for_computer_vision.pdf – Section 2.2 "Color transformation"

#### 5.3.2 Normalizing Pixel Values

    images_pp = preprocess_input(images_rgb.astype(np.float32))

What preprocess_input does:
- Converts pixel values from [0, 255] to [-1, 1]
- This matches the normalization used during MobileNetV2 training
- Important: using the wrong normalization significantly reduces accuracy

### 5.4 Step 3: Loading and Freezing MobileNetV2

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

Why include_top=False:
- Excludes the final classification layers (which output 1000 ImageNet classes)
- We only need the feature extraction layers

Why freeze (trainable = False):
- Preserves the general features learned on ImageNet
- Prevents overfitting (we have only 208 images)
- Speeds up computation (no weight updates)

Why freezing works:
- The model already knows how to detect edges, textures, and patterns
- These universal features are exactly what we need for molecular structures
- Only the final layers should be trained on new data

### 5.5 Step 4: Global Average Pooling (The Key Challenge)

#### 5.5.1 The Problem

In the guide, it said to extract base_model.get_layer('global_average_pooling2d').output. However, when we tried this:

    ValueError: No such layer: global_average_pooling2d

Why this happened:
- In the version of MobileNetV2 loaded from TensorFlow, the final convolutional layer is named out_relu
- The Global Average Pooling is not explicitly named global_average_pooling2d in this version

#### 5.5.2 The Solution

Instead of looking for the layer by name, we manually added the pooling layer:

    from tensorflow.keras.layers import GlobalAveragePooling2D

    gap_layer = GlobalAveragePooling2D()
    embeddings_output = gap_layer(base_model.output)

    embedding_model = Model(
        inputs=base_model.input,
        outputs=embeddings_output
    )

Why this works:
- base_model.output is the output of the last convolutional layer (out_relu)
- The shape is (None, 7, 7, 1280) – still spatial
- The Global Average Pooling compresses 7×7×1280 to 1280

What is Global Average Pooling:
- Takes each feature map (7×7) and averages it to a single number
- For 1280 feature maps → 1280 numbers
- Each number represents the "amount" of a particular learned feature present in the image

Reference: 02-intro_to_computer_vision_pytorch.pdf – Section 3: "Building a Simple CNN"

### 5.6 Step 5: Extracting Embeddings

    embeddings = embedding_model.predict(images_pp, batch_size=32, verbose=1)
    # Output shape: (208, 1280)

What happens here:
- Each image is passed through the frozen MobileNetV2
- The output is compressed to a 1280-dimensional vector
- The result is a matrix of shape (208, 1280)

### 5.7 Step 6: Saving Embeddings

    np.save(OUTPUT_EMBED, embeddings)
    meta.to_csv(OUTPUT_META, index=False)

Why .npy format:
- Fast to load and save
- Compatible with PyTorch, Keras, scikit-learn
- Preserves the exact NumPy array structure

---

## 6. Dealing with TensorFlow Warnings

### 6.1 The oneDNN Warnings

When running the script, we saw:

    I tensorflow/core/util/port.cc:153] oneDNN custom operations are on...

What this means:
- oneDNN is Intel's library for optimizing math operations on CPU
- TensorFlow is telling us it's using these optimizations
- It's not an error – just an informational message

### 6.2 How to Disable Them (Optional)

    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

Note: This must be placed before any import tensorflow statements.

### 6.3 The CPU Feature Guard Warning

    I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions...

What this means:
- TensorFlow is informing us that it's using SSE, AVX, etc. for better performance
- Not an error – it's informational

Solution: Ignore it, or use the os.environ approach to suppress all logs.

---

## 7. Key Decisions and Their Rationale

| Decision | Rationale |
|----------|-----------|
| Use MobileNetV2 instead of ResNet50 | Lighter, faster, sufficient for our small dataset |
| Freeze all layers | Preserve ImageNet features, prevent overfitting |
| Convert grayscale to RGB by repeating channel | Make images compatible with MobileNetV2 |
| Use preprocess_input | Ensure correct normalization for MobileNetV2 |
| Add Global Average Pooling manually | Because the layer name wasn't as expected in this version |
| Save as .npy | Fast loading, compatible with downstream tools |

---

## 8. Obstacles Encountered and Resolutions

| Obstacle | Resolution |
|----------|------------|
| global_average_pooling2d not found | Added GlobalAveragePooling2D manually using base_model.output |
| TensorFlow warnings cluttering output | Set os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' |
| Metadata path mismatch | Moved metadata from phase1/images/ to phase1/data/ |
| Model output shape was (None, 7, 7, 1280) | Applied Global Average Pooling manually |

---

## 9. Final Results and Deliverables

### 9.1 Output Files

| File | Content | Shape |
|------|---------|-------|
| phase2/data/embeddings.npy | Visual embeddings | (208, 1280) |
| phase2/data/embeddings_metadata.csv | Metadata: name, smiles, role, augment | 208 rows |

### 9.2 Validation

    import numpy as np
    embeddings = np.load('phase2/data/embeddings.npy')
    print(embeddings.shape)  # (208, 1280)
    print(embeddings.mean())  # ~0.0 (normalized)
    print(embeddings.std())   # ~1.0 (normalized)

### 9.3 What We Learned

1. Transfer learning is powerful – We used a model trained on natural images to extract features from molecular images.
2. Preprocessing matters – The correct normalization is critical for model performance.
3. Layer names can vary – Always verify the actual layer names in your version of a pre-trained model.
4. Metadata is essential – We preserved traceability between embeddings and their sources.

---

## 10. Relationship to Other Pipeline Phases

### Phase 1 (Completed)

    [Molecule Names] 
        → fetch_molecules.py → [SMILES CSV]
        → generate_images.py → [Images NPZ] + [Metadata CSV]

### Phase 2 (Completed Now)

    [Images NPZ] 
        → extract_embeddings.py → [Embeddings NPY] + [Metadata CSV]

### Phase 3 (Next Step)

    [Embeddings NPY] + [Environmental Features]
        → Regressor (PyTorch/XGBoost)
        → Predict %Curing Conversion

---

## 11. Complete Script Code

For reference, here is the final extract_embeddings.py script:

    import os
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from tensorflow.keras import Model
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.layers import GlobalAveragePooling2D

    # ==================== CONFIGURATION ====================

    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    IMAGES_P1_DIR = BASE_DIR.parent / "phase1" / "images"
    DATA_P1_DIR = BASE_DIR.parent / "phase1" / "data"

    INPUT_NPZ = IMAGES_P1_DIR / "molecular_images.npz"
    INPUT_META = DATA_P1_DIR / "molecular_metadata.csv"
    OUTPUT_EMBED = DATA_DIR / "embeddings.npy"
    OUTPUT_META = DATA_DIR / "embeddings_metadata.csv"

    # ==================== LOAD IMAGES ====================

    data = np.load(INPUT_NPZ)
    images = data['images']
    meta = pd.read_csv(INPUT_META)

    print(f'{len(images)} images loaded.')
    print(f'    Shape of an image: {images[0].shape}')
    print(f'    Metadata: {len(meta)} rows')
    print(f'    Columns: {meta.columns.tolist()}')

    # ==================== PREPROCESS ====================

    print("Images preprocessing started...")
    images_rgb = np.stack([images] * 3, axis=-1)
    print(f'  RGB shape: {images_rgb.shape}')

    images_pp = preprocess_input(images_rgb.astype(np.float32))
    print(f'  Preprocessed shape: {images_pp.shape}')
    print(f'  Min value: {images_pp.min():.2f}, Max value: {images_pp.max():.2f}')

    # ==================== LOAD MODEL ====================

    print("Loading of MobileNetV2...")
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    print(f'Model successfully loaded and frozen.')
    print(f'  Total layers: {len(base_model.layers)}')
    print(f'  Trainable layers: {sum(1 for layer in base_model.layers if layer.trainable)}')

    # Add Global Average Pooling manually
    gap_layer = GlobalAveragePooling2D()
    embeddings_output = gap_layer(base_model.output)

    embedding_model = Model(
        inputs=base_model.input,
        outputs=embeddings_output
    )

    print("Embedding model successfully created.")
    print(f'  Output shape: {embedding_model.output_shape}')  # (None, 1280)

    # ==================== EXTRACT EMBEDDINGS ====================

    print("Extracting embeddings...")
    embeddings = embedding_model.predict(images_pp, batch_size=32, verbose=1)
    print(f'Embeddings successfully extracted.')
    print(f'  Shape: {embeddings.shape}')

    # ==================== SAVE ====================

    np.save(OUTPUT_EMBED, embeddings)
    meta.to_csv(OUTPUT_META, index=False)

    print(f'\nEmbeddings saved to {OUTPUT_EMBED}')
    print(f'\nMetadata saved to {OUTPUT_META}')

---

## 12. Conclusion

extract_embeddings.py successfully completes Phase 2 of the MVP pipeline, transforming molecular images into dense vector representations. The script:

- Implements the specification exactly as described
- Uses transfer learning (MobileNetV2 with frozen weights)
- Extracts 1280-dimensional embeddings from 208 images
- Saves embeddings in efficient .npy format
- Preserves metadata for traceability
