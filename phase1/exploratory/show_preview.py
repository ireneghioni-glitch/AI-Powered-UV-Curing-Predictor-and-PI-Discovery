import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
IMAGES_DIR = Path(__file__).resolve().parent / "images"

# Carica le immagini
data = np.load(IMAGES_DIR / "molecular_images.npz")
images = data['images']

# Carica i metadati (nota: è in DATA_DIR, non IMAGES_DIR)
meta = pd.read_csv(DATA_DIR / "molecular_metadata.csv")

# Mostra le prime 4 immagini
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    axes[i].imshow(images[i], cmap='gray')
    axes[i].set_title(f"{meta.iloc[i]['name']} {meta.iloc[i]['augment']}")
    axes[i].axis('off')
plt.tight_layout()
plt.savefig(DATA_DIR / "preview_new.png", dpi=100)
print(f"✅ Preview salvata in {DATA_DIR / 'preview_new.png'}")
plt.show()