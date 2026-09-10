# MVP Operational Guide: AI-Powered UV-Curing Predictor & PI Discovery

This guide serves as the technical development and study manual to build your MVP (Minimum Viable Product) in less than two weeks. It directly connects the theoretical concepts from your study materials (Computer Vision, Deep Learning, and Machine Learning) with the practical implementation of the chemical project and the deployment of the reactive web application in pure Python using **Reflex**.

---

## 🗺️ Conceptual Workflow Map (Operational Sequence)

The project is divided into 5 consecutive logical phases. Each phase maps to a specific module in your educational materials:

```
[Phase 1: Data Preparation] (OpenCV, RDKit, PubChem API)
             │
             ▼
[Phase 2: Computer Vision] (CNN Transfer Learning, Images as Tensors)
             │
             ▼
[Phase 3: Deep Learning Core] (PyTorch / Keras, Training Loop, Gradient Descent)
             │
             ▼
[Phase 4: Integration & ML] (XGBoost / Regression, Mixture Engineering)
             │
             ▼
[Phase 5: Reactive Web App] (Reflex Backend, State, SQLModel Database, Auth)
```

---

## 🧪 Phase 1: Data Ingestion and Chemical Preprocessing
**What you learn in your materials (CV - Module 00):** 
For a computer, images are three-dimensional matrices (Height, Width, RGB Channels) with values from 0 to 255. Preprocessing standardizes formats and removes noise, reducing computational complexity.

### Implementation in the Project
Instead of working with lung images (Pneumonia Challenge), your "Chemical Vision" works with the two-dimensional structures of monomers and photoinitiators (PI).

1. **SMILES Ingestion**: Load known molecules (e.g., Benzophenone: `O=C(c1ccccc1)c2ccccc2`) using the **RDKit** cheminformatics library.
2. **Image Generation**: RDKit converts the SMILES text string into a 2D pixel grid (black and white vector drawing) of a fixed size, for example, $224 \times 224$ pixels.
3. **Channel Optimization (Color Transformation)**: As described in module 2.2 of CV preprocessing, object classification relies primarily on shapes rather than colors. Convert the molecular drawings to **grayscale (1 channel instead of 3 RGB)**, reducing the memory footprint by 66% without losing structural information (bonds or atoms).
4. **Data Augmentation**: In module 2.3, you learn to apply rotations and mirroring using OpenCV to double or quadruple the dataset size. Apply controlled rotations ($90^\circ, 180^\circ$) on the RDKit drawings to simulate spatial orientation independence of the molecule.

```python
# Example of Chemical Preprocessing with RDKit
from rdkit import Chem
from rdkit.Chem import Draw
import cv2
import numpy as np

def smiles_to_grayscale_image(smiles, size=(224, 224)):
    # SMILES -> Molecule Object conversion
    mol = Chem.MolFromSmiles(smiles)
    # 2D drawing generation (default RGB)
    img = Draw.MolToImage(mol, size=size)
    img_np = np.array(img)
    
    # Grayscale conversion (OpenCV)
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return gray_img
```

---

## 👁️ Phase 2: Computer Vision and Feature Extraction
**What you learn in your materials (CV - Modules 01, 04, 05):**
*Transfer Learning* consists of importing a pre-trained model trained on millions of images (such as ImageNet), freezing its convolutional weights (which already know how to recognize lines, edges, and geometric shapes), and using it as a feature extractor.

### Implementation in the Project
Pre-trained models in Keras or PyTorch (e.g., ResNet50 or MobileNetV2) are perfect for recognizing the functional groups in your molecular drawings.

1. **Model Import**: Load a lightweight convolutional network (MobileNetV2 is ideal for fast computation on CPU).
2. **Layer Freezing**: Set `trainable = False` on all convolutional layers of the network.
3. **Visual Embeddings Generation**: Pass the grayscale molecular drawing through the network. Extract the output of the penultimate layer (Global Average Pooling). You will obtain a dense vector (e.g., of 1024 numbers) that mathematically describes the "geometry" of the molecule. The network will associate similar values with molecules containing similar structures (e.g., the aromatic ring of Benzophenone and TPO).

---

## 🧠 Phase 3: The Deep Learning Core and the Training Loop
**What you learn in your materials (DL - Modules 02, 03, 04):**
An artificial neuron (Perceptron) performs a weighted sum of inputs plus a bias: $z = \sum x_i w_i + b$. The value passes through an activation function (ReLU for hidden layers, Sigmoid or Softmax for output) to introduce non-linearity. During the *Backward Pass*, the error (Loss) is backpropagated using gradient descent to update the weights via the optimizer (e.g., Adam).

### Project Implementation (Tutorial for your Training Loop)
Here is how to translate the theory of your modules into a functioning training loop in **PyTorch** to map the embeddings extracted from the CNN and compute the prediction error:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Defining the Regressor architecture (Multilayer Perceptron)
class CuringPredictorNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Linear structure with ReLU as per chapter 4 of your PDFs
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1) # Continuous output for the conversion % (Regression)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)

# Model, loss function, and optimizer configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CuringPredictorNet(input_dim=1024).to(device)
criterion = nn.MSELoss() # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# The Training Loop (Guess -> Check -> Tweak)
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 1. Zero out accumulated gradients
        optimizer.zero_grad()
        # 2. Forward pass (make prediction)
        outputs = model(inputs)
        # 3. Compute error (Loss)
        loss = criterion(outputs, targets)
        # 4. Backward pass (backpropagate the error)
        loss.backward()
        # 5. Optimization (update weights of the artificial brain)
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)
```

---

## 📊 Phase 4: Tabular Integration & Machine Learning (XGBoost)
**What you learn in your materials (Data Analysis - "Time Series & Regression" Module):**
Many real-world problems can be solved by converting them into tabular regression tasks. With libraries like **PyCaret** or **XGBoost**, we can map complex non-linear relationships starting from data in row/column format.

### Implementation in the Project
The final curing conversion percentage ($Y$) does not only depend on the identity of the molecule, but also on the recipe and process parameters.

1. **Mixture Feature Engineering (Concatenation)**:\
   Merge the visual embeddings extracted by the CNN of your monomers and photoiniziators with the physico-chemical parameters and operational variables into a single tabular vector:\
   $$\text{Row} = [\text{Embed Monomer}] + [\text{Embed PI}] + [\text{Is\_Aqueous}] + [\text{LogP}] + [\% \text{ PI}] + [\text{Dose UV}]$$
2. **Training with XGBoost**:\
   This tabular input vector is fed into an XGBoost Regressor model (easy to set up and extremely powerful for tabular data) to predict the continuous target `Double_Bond_Conversion_Percentage` (from 0 to 100%).

---

## ⚡ Phase 5: Interactive Deployment with Reflex (Replacing Streamlit)
**What you learn in your materials (Reflex YouTube Course):**
Reflex allows you to create full-stack, responsive, and modern web applications writing exclusively Python code (without JavaScript, HTML, or CSS). Reflex is built on three pillars:
* **State**: Python classes inheriting from `rx.State` containing instance variables (vars) and methods (event handlers) that modify the state reactively.
* **UI Components**: Ready-made elements (`rx.heading`, `rx.input`, `rx.hstack`, `rx.form`) written in Python that compile to React.
* **Integrated Database**: Managed natively via SQLModel (`rx.model`) integrated with automatic migrations via Alembic.

### 🛠️ Step-by-Step Tutorial for Reflex in your MVP

#### 1. Project Initialization
Open the terminal, activate your virtual environment, and run:
```bash
pip install reflex reflex-local-auth
reflex init
```
Choose option `0` (Blank template) to have a clean structure.

#### 2. Database Model Definition (`rx.model`)
In the `models.py` file (or inside the UI/backend folder of your Reflex project), define the schema to save the curing tests performed by users:

```python
import reflex as rx
from datetime import datetime, timezone

class CuringLog(rx.Model, table=True):
    monomer_smiles: str
    pi_smiles: str
    environment: str # "Aqueous" or "Solvent"
    uv_dose: float # mJ/cm²
    predicted_conversion: float # % calculated by the model
    created_at: datetime = datetime.now(timezone.utc)
```
Run the migration of the local SQLite database to create the table:
```bash
reflex db init
reflex db make-migrations
reflex db migrate
```

#### 3. Reactive State and Model Management (`rx.State`)
In `state.py`, define the state class that manages the prediction of curing conversion (XGBoost + RDKit) and interaction with the database:

```python
import reflex as rx
from .models import CuringLog

class PredictorState(rx.State):
    # Reactive UI variables
    monomer_smiles: str = "C=CC(=O)O" # Default: Acrylic Acid
    pi_smiles: str = "O=C(c1ccccc1)c2ccccc2" # Default: Benzophenone
    environment: str = "Solvent"
    uv_dose: float = 150.0
    
    # Prediction results
    predicted_conversion: float = 0.0
    is_loading: bool = False

    # Asynchronous Event Handler to simulate computation and perform the prediction
    @rx.event
    async def handle_prediction(self, form_data: dict):
        self.is_loading = True
        yield # Yield forces an immediate UI update (shows the spinner)
        
        # Extracting data from the Reflex form
        self.monomer_smiles = form_data.get("monomer")
        self.pi_smiles = form_data.get("pi")
        self.environment = form_data.get("env")
        self.uv_dose = float(form_data.get("dose", 150.0))
        
        # 1. Execution of the Chemical Vision pipeline (RDKit + CNN Embedding) -> Simulated here
        # In production: embed = cnn.predict(smiles_to_img(self.pi_smiles))
        
        # 2. Prediction with XGBoost -> Simulated with a physical equation for the MVP
        # (Replace this block with model.predict() once you have trained the XGBoost)
        import math
        solubility = 0.95 if self.environment == "Solvent" else 0.20 # The PI does not dissolve in water!
        raw_pred = 100 * (1 - math.exp(-0.01 * self.uv_dose)) * solubility
        self.predicted_conversion = round(min(max(raw_pred, 0.0), 100.0), 2)
        
        # 3. Save prediction in the SQLite database via Reflex Session
        with rx.session() as session:
            log_entry = CuringLog(
                monomer_smiles=self.monomer_smiles,
                pi_smiles=self.pi_smiles,
                environment=self.environment,
                uv_dose=self.uv_dose,
                predicted_conversion=self.predicted_conversion
            )
            session.add(log_entry)
            session.commit()
            
        self.is_loading = False
        yield
```

#### 4. Building the Reactive Graphical Interface (UI)
In your main file (e.g., `app.py` or `pages/index.py`), write the layout in pure Python leveraging Reflex components:

```python
import reflex as rx
from .state import PredictorState

def index() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.heading("AI-Powered UV-Curing Predictor 🧪", size="8", margin_y="4"),
            rx.text(
                "Enter molecular structures and curing parameters to compute the theoretical double bond conversion.",
                size="4", color_scheme="gray"
            ),
            
            # Data entry form
            rx.form(
                rx.vstack(
                    rx.text("Monomer SMILES:", font_weight="bold"),
                    rx.input(name="monomer", default_value=PredictorState.monomer_smiles, width="100%"),
                    
                    rx.text("Photoinitiator SMILES:", font_weight="bold"),
                    rx.input(name="pi", default_value=PredictorState.pi_smiles, width="100%"),
                    
                    rx.hstack(
                        rx.box(
                            rx.text("Environment:", font_weight="bold"),
                            rx.select(["Solvent", "Aqueous"], name="env", default_value=PredictorState.environment),
                            width="50%"
                        ),
                        rx.box(
                            rx.text("UV Energy Dose (mJ/cm²):", font_weight="bold"),
                            rx.input(name="dose", type="number", default_value=str(PredictorState.uv_dose)),
                            width="50%"
                        ),
                        width="100%", spacing="4"
                    ),
                    
                    rx.button(
                        rx.cond(PredictorState.is_loading, rx.spinner(), "Calculate Curing Conversion ⚡"),
                        type="submit", color_scheme="sky", width="100%", margin_top="4"
                    ),
                    spacing="3", width="100%"
                ),
                on_submit=PredictorState.handle_prediction,
                width="600px", border="1px solid #ccc", padding="6", border_radius="lg", bg="white"
            ),
            
            # Reactive Condition: show result if computed
            rx.cond(
                PredictorState.predicted_conversion > 0.0,
                rx.vstack(
                    rx.heading("Prediction Result 📊", size="5", margin_top="6"),
                    rx.box(
                        rx.hstack(
                            rx.text("Curing Conversion Percentage:", font_weight="semibold"),
                            rx.text(f"{PredictorState.predicted_conversion}%", font_weight="bold", color="green" if PredictorState.predicted_conversion > 75 else "orange"),
                            spacing="4"
                        ),
                        border="2px solid green" if PredictorState.predicted_conversion > 75 else "2px solid orange",
                        padding="4", border_radius="md", bg="#f9fff9" if PredictorState.predicted_conversion > 75 else "#fffbf2",
                        width="100%"
                    ),
                    width="600px"
                )
            ),
            align="center", spacing="5", padding_y="10"
        ),
        size="3"
    )

app = rx.App()
app.add_page(index, route="/")
```

---

## 🏁 MVP Closing Checklist (Under 2 Weeks)
By following this operational integration, your GitHub repository will contain:
- [ ] **`/backend`**: Python scripts with the RDKit pipeline and the PyTorch/XGBoost model class.
- [ ] **`/app`**: Reflex code with the reactive prediction `State` and the application UI.
- [ ] **`db.sqlite`**: Auto-generated local SQLite database with the schema of the executed predictions.
- [ ] **`README.md`**: Detailed explanation of curing physics, databases used for training, and instructions to run the local app with `reflex run`.

This "theoretically finished" and flawless MVP will demonstrate your excellent ability to merge Computer Vision and reactive Software Engineering into a single, formidable industrial project.
