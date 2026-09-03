# Phase 2: Computer Vision and Feature Extraction – Detailed Step‑by‑Step Guide

This guide walks you through the second phase of your MVP pipeline. By the end of this phase you will have extracted dense vector representations (embeddings) from your molecular images using a pre‑trained Convolutional Neural Network (CNN). These embeddings will be used in Phase 3 as input to the regression model.

You will learn how to:
* Load the molecular images from the NPZ file created in Phase 1.
* Understand what CNNs are and why we use transfer learning.
* Preprocess images to match the requirements of a pre‑trained CNN.
* Use a pre‑trained model (MobileNetV2) as a feature extractor.
* Extract the penultimate layer’s output (Global Average Pooling) to obtain embeddings.
* Save the embeddings as a NumPy array for the next phase.

---

## 1. Theoretical Foundations: What is a Convolutional Neural Network (CNN)?

### 1.1 From Pixels to Patterns

A computer “sees” an image as a grid of numbers. A grayscale image is a 2D tensor of shape (Height, Width). A color image is a 3D tensor of shape (Channels, Height, Width) in PyTorch's convention.

Reference: `02-intro_to_computer_vision_pytorch.pdf` – Section 1: “Images as Tensors”

A CNN is a special type of neural network that is designed to process data with a grid‑like topology, such as images. It learns to detect patterns by applying small filters (kernels) that slide over the image.

Reference: `03-the-deep-in-deep-learning.pdf` – Section on Forward Pass

### 1.2 CNN Architecture Layers

A typical CNN consists of:

| **Layer** | **Purpose** |
|-----------|-------------|
| **Convolutional Layer** | Applies learnable filters to detect local patterns (edges, textures, shapes). |
| **Activation Layer (ReLU)** | Introduces non‑linearity so the network can learn complex representations. |
| **Pooling Layer** | Reduces spatial dimensions, summarises features, and decreases computation. |
| **Flatten Layer** | Converts the 3D feature map into a 1D vector for the fully connected layers. |
| **Fully Connected (Dense) Layer** | Performs classification or regression based on extracted features. |

Reference: `15-minute - Women TechMaker Brussel - Slide Deck.pdf` – Slides 15‑22

### 1.3 Why CNNs Work So Well for Images

- **Local Connectivity**: Each neuron connects only to a small region of the input (receptive field).
- **Parameter Sharing**: The same filter is applied across the entire image, drastically reducing the number of parameters.
- **Hierarchical Feature Learning**: Early layers detect simple features (edges, corners), while deeper layers combine them into complex structures (eyes, wheels, functional groups).

Reference: `01-image-classification-theory.md` – Section “CNN Image Classification”

### 1.4 Transfer Learning: Why We Don’t Train from Scratch

Training a CNN from scratch requires:

- Millions of labelled images (e.g., ImageNet has 14 million images).
- Massive computational resources (GPUs for days or weeks).
- Expertise in hyperparameter tuning and architecture design.

**Transfer learning** solves this by taking a model already pre‑trained on a large dataset (e.g., ImageNet) and reusing its learned features for a new, similar task.

Reference: `15-minute - Women TechMaker Brussel - Slide Deck.pdf` – Slides 28‑31

**Why it works**: The early layers of a CNN learn general‑purpose features (edges, textures, simple shapes) that are useful for almost any image recognition task. Only the later layers learn task‑specific patterns.

**Our approach**:
1. Load a pre‑trained CNN (MobileNetV2) with ImageNet weights.
2. **Freeze** all convolutional layers (prevent them from being updated during training).
3. Use the frozen model as a fixed **feature extractor**.
4. Pass each molecular image through the network and extract the output of the **Global Average Pooling** layer – this is our embedding.

---

## 2. Why MobileNetV2?

| **Advantage** | **Reason** |
|---------------|------------|
| **Lightweight** | Designed for mobile and embedded vision applications; runs fast on CPU. |
| **Pre‑trained on ImageNet** | Has learned to recognise edges, textures, and shapes – useful for molecular structures. |
| **Global Average Pooling output** | Produces a dense vector of 1280 features per image – a manageable size for our regressor. |
| **Widely supported** | Available in both PyTorch (torchvision) and Keras (tensorflow.keras.applications). |

Reference: `curing-prediction-pipeline-mvp-v3-en.md` – Phase 2: “Model Import”

---

## 3. Step‑by‑Step Implementation

### Step 3.1: Environment Setup

Ensure you have the necessary libraries installed. In your `chemvision` environment:

```bash
pip install tensorflow numpy pandas matplotlib
```

or if you prefer PyTorch:

```bash
pip install torch torchvision numpy pandas matplotlib
```

> **Note**: The example in the specification uses Keras, but the concepts are identical in PyTorch.

### Step 3.2: Load the Molecular Images

We will load the NPZ file generated in Phase 1.

```python
import numpy as np
import pandas as pd

# Load images and metadata
data = np.load('phase1/images/molecular_images.npz')
images = data['images']  # shape: (208, 224, 224)
meta = pd.read_csv('phase1/images/molecular_metadata.csv')

print(f"Loaded {len(images)} images of shape {images[0].shape}")
```

**Key concept**: The images are currently grayscale (1 channel, 224×224). Most pre‑trained CNNs expect 3‑channel RGB images. We need to convert them.

### Step 3.3: Preprocess Images for the CNN

Pre‑trained models expect input in a specific format. For MobileNetV2 in Keras:

- **Dimensions**: 224 × 224 pixels.
- **Channels**: 3 (RGB). We need to convert grayscale (1 channel) to RGB by repeating the same channel 3 times.
- **Data type**: float32.
- **Pixel range**: 0‑1 (normalisation) – or use the model’s own preprocessing function.
- **Batch dimension**: add an extra axis for batch processing.

Reference: `00-preprocessing_for_computer_vision.pdf` – Section 2.1 “Image resizing” and Section 2.2 “Color transformation”

**Step 3.3.1: Convert Grayscale to RGB**

```python
# Convert (N, 224, 224) to (N, 224, 224, 3)
images_rgb = np.stack([images] * 3, axis=-1)
print(images_rgb.shape)  # (208, 224, 224, 3)
```

**Step 3.3.2: Normalise Pixel Values**

MobileNetV2 expects pixel values in the range `[-1, 1]`. The `tf.keras.applications.mobilenet_v2.preprocess_input` function does this automatically.

```python
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Convert to float32 and apply preprocessing
images_preprocessed = preprocess_input(images_rgb.astype(np.float32))
```

### Step 3.4: Load Pre‑trained MobileNetV2 and Freeze Weights

We load the model with ImageNet weights and set `include_top=False` to exclude the final classification layers. We then freeze all layers so they are not updated.

```python
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model

# Load model without the top classification layers
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze all layers
base_model.trainable = False

# Print summary to see the architecture
base_model.summary()
```

**Why we freeze**: We want to use the pre‑learned features without modifying them. The model already knows how to detect edges, textures, and patterns – exactly what we need for molecular structures.

Reference: `04-object_detection_pytorch_example.pdf` – Part 2: “Fine‑tuning on a custom dataset”

### Step 3.5: Extract Embeddings (Feature Vectors)

We pass each image through the frozen model and extract the output of the **Global Average Pooling** layer. In MobileNetV2, this is the layer named `global_average_pooling2d`.

```python
# Create a new model that outputs the embeddings
embedding_model = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer('global_average_pooling2d').output
)

# Predict embeddings for all images
embeddings = embedding_model.predict(images_preprocessed)

print(embeddings.shape)  # (208, 1280)
```

**Explanation**:
- The `global_average_pooling2d` layer takes the feature maps (7×7×1280) and averages each feature map across the spatial dimensions, producing a vector of 1280 numbers.
- Each number represents the “amount” of a particular learned feature present in the image.
- This vector is a dense, lower‑dimensional representation (embedding) of the molecular structure.

Reference: `02-intro_to_computer_vision_pytorch.pdf` – Section 3: “Building a Simple CNN”

### Step 3.6: Save Embeddings for Phase 3

We save the embeddings as a NumPy array (`.npy`) for use in the regression phase.

```python
# Save embeddings
np.save('phase1/images/embeddings.npy', embeddings)

# Also save the corresponding metadata (name, role, augment) for traceability
meta.to_csv('phase1/images/embeddings_metadata.csv', index=False)

print(f"Saved embeddings of shape {embeddings.shape}")
```

**Why this format**: NumPy’s `.npy` format is fast to load and compatible with PyTorch, Keras, and scikit‑learn, which we will use in later phases.

---

## 4. What Are Embeddings and Why Do We Use Them?

### 4.1 The Problem with Raw Pixels

Raw pixels are:
- **High‑dimensional**: 224×224×3 = 150,528 numbers per image.
- **Redundant**: Adjacent pixels are highly correlated.
- **Lacking semantic meaning**: The raw values don’t tell us “this is an aromatic ring” or “this is a carbonyl group”.

### 4.2 What Embeddings Do

A CNN compresses the raw pixels into a much smaller, dense vector (e.g., 1280 numbers) that captures the semantic content of the image.

- **Rich features**: Each number represents the presence of a specific learned pattern.
- **Reduced dimension**: From 150,528 to 1280 – a 99% reduction.
- **Better for regression**: The embedding vectors are more linearly separable and easier for a regressor (XGBoost) to model.

Reference: `03-the-deep-in-deep-learning.pdf` – Section on “Forward pass” and “How does a neural network learn”

### 4.3 How the CNN Learns These Features

During training on ImageNet, the CNN learns to detect:
- **Early layers**: simple edges, colours, and textures.
- **Middle layers**: combinations of edges (corners, curves, simple shapes).
- **Late layers**: complex objects (eyes, wheels, faces).

For molecular images, the same principles apply:
- Early layers detect bond angles and ring structures.
- Middle layers detect functional groups (carbonyl, hydroxyl, aromatic rings).
- Late layers combine these into whole molecular “fingerprints”.

Reference: `01-image-classification-theory.md` – Section “CNN Image Classification”

---

## 5. Summary of Phase 2 Deliverables

After completing this phase, your project folder should contain:

| **File** | **Description** |
|----------|-----------------|
| `phase1/images/embeddings.npy` | NumPy array of shape (208, 1280) – the visual embeddings. |
| `phase1/images/embeddings_metadata.csv` | Metadata linking each embedding to its molecule name, role, and augmentation type. |

You have also learned:
- What CNNs are and how they work.
- Why transfer learning is powerful and efficient.
- How to preprocess images for a pre‑trained CNN.
- How to extract embeddings using a frozen model.

---

## 6. Next Steps (Preview of Phase 3)

Now that we have embeddings, the next phase will:

1. **Load the embeddings** and the corresponding metadata.
2. **Pair each embedding with the environmental features** (Is_Aqueous, LogP, PI_Concentration, UV_Dose).
3. **Train a regressor** (PyTorch neural network or XGBoost) to predict the %Curing Conversion.

This is where the “Deep Learning Core” of the pipeline comes into play, as described in the specification.

Reference: `curing-prediction-pipeline-mvp-v3-en.md` – Phase 3: “The Deep Learning Core and the Training Loop”

---

## 7. Theoretical Summary of Key Concepts

### 7.1 Convolutional Neural Networks (CNNs)

A CNN processes images by applying a series of transformations:

```text
Input Image → Conv + ReLU → Pool → Conv + ReLU → Pool → Flatten → Dense → Output
```

Each layer builds a more abstract representation of the input.

### 7.2 Transfer Learning

We use a model pre‑trained on ImageNet (1.4 million images, 1000 classes). By freezing its weights, we preserve the general visual features it has learned, and we only use it as a feature extractor.

### 7.3 Feature Extraction vs. Fine‑Tuning

| **Approach** | **Description** | **When to use** |
|--------------|-----------------|-----------------|
| **Feature Extraction** | Freeze the entire model; use it as a fixed feature extractor. | Small dataset; low computational resources. |
| **Fine‑Tuning** | Unfreeze some of the top layers and retrain them on your dataset. | Larger dataset; specialised domain; need to adapt features. |

For our MVP, feature extraction is sufficient.

Reference: `04-object_detection_pytorch_example.pdf` – Part 2: “Build the model: swap the head, keep the pretrained backbone”

---

## 8. Complete Code for Phase 2

Below is the complete script to run Phase 2. Save it as `phase2_extract_embeddings.py`.

```python
import numpy as np
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import Model
from pathlib import Path

# ==================== CONFIGURATION ====================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "phase1" / "images"

INPUT_NPZ = DATA_DIR / "molecular_images.npz"
INPUT_META = DATA_DIR / "molecular_metadata.csv"
OUTPUT_EMBED = DATA_DIR / "embeddings.npy"
OUTPUT_META = DATA_DIR / "embeddings_metadata.csv"

# ==================== LOAD IMAGES ====================

print("Loading images...")
data = np.load(INPUT_NPZ)
images = data['images']  # shape: (N, 224, 224)
meta = pd.read_csv(INPUT_META)
print(f"Loaded {len(images)} images.")

# ==================== PREPROCESS ====================

print("Preprocessing images...")
# Convert grayscale to RGB by repeating the channel
images_rgb = np.stack([images] * 3, axis=-1)
# Convert to float32 and apply MobileNetV2 preprocessing
images_pp = preprocess_input(images_rgb.astype(np.float32))
print(f"Preprocessed shape: {images_pp.shape}")

# ==================== LOAD MODEL ====================

print("Loading MobileNetV2...")
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False  # Freeze all layers

# Create embedding model
embedding_model = Model(
    inputs=base_model.input,
    outputs=base_model.get_layer('global_average_pooling2d').output
)
print("Model loaded.")

# ==================== EXTRACT EMBEDDINGS ====================

print("Extracting embeddings...")
embeddings = embedding_model.predict(images_pp, batch_size=32, verbose=1)
print(f"Embeddings shape: {embeddings.shape}")

# ==================== SAVE ====================

np.save(OUTPUT_EMBED, embeddings)
meta.to_csv(OUTPUT_META, index=False)

print(f"\n✅ Embeddings saved to {OUTPUT_EMBED}")
print(f"✅ Metadata saved to {OUTPUT_META}")
```

---

## 9. Testing and Validation

### 9.1 Verify the Embeddings

After running the script, you can verify the output:

```python
import numpy as np

embeddings = np.load('phase1/images/embeddings.npy')
print(f"Embeddings shape: {embeddings.shape}")  # Expected: (208, 1280)
print(f"Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")
```

### 9.2 Visualise the Embeddings (Optional)

You can use PCA or t‑SNE to project the 1280‑dimensional embeddings into 2D for visual inspection.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce to 2D
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Plot by role
meta = pd.read_csv('phase1/images/embeddings_metadata.csv')
colors = {'PI_TypeI': 'red', 'PI_TypeII': 'blue', 'co-initiator': 'green'}

plt.figure(figsize=(10, 8))
for role, color in colors.items():
    mask = meta['role'] == role
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                c=color, label=role, alpha=0.7)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Embeddings Visualised with PCA')
plt.legend()
plt.savefig('phase1/images/embeddings_pca.png', dpi=100)
plt.show()
```

---

## 10. Troubleshooting & Tips

| **Issue** | **Solution** |
|-----------|--------------|
| **Keras/TensorFlow not installed** | `pip install tensorflow` |
| **Out of memory** | Reduce batch size in `model.predict()` (e.g., `batch_size=16`) |
| **Images not loading** | Check that the path to `molecular_images.npz` is correct. |
| **Embeddings shape unexpected** | Check that you used `include_top=False` and extracted the correct layer. |
| **Slow prediction** | Use a smaller batch size or run on GPU (if available). |

---

## 11. Summary

You have now completed Phase 2:

- ✅ Learned the theoretical foundations of CNNs and transfer learning.
- ✅ Loaded and preprocessed molecular images for the CNN.
- ✅ Used MobileNetV2 as a frozen feature extractor.
- ✅ Extracted embeddings of shape (208, 1280).
- ✅ Saved embeddings for use in Phase 3.

**You are now ready for Phase 3: The Deep Learning Core.**
