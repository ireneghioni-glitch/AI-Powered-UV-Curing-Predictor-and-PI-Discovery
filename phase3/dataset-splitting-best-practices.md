# Best Practices for Dataset Splitting: Train, Validation, and Test Sets

In machine learning and deep learning pipelines, splitting data correctly is fundamental to building models that generalize well to unseen real-world scenarios [205]. This guide outlines the differences between the three dataset partitions, explains how to implement a three-way split in PyTorch, and discusses when this practice is critical.

---

## 1. The Three Datasets and Their Purposes

A standard, rigorous machine learning workflow divides the total available dataset into three distinct, non-overlapping subsets [205]:

| Dataset | Core Purpose | When and How It Is Used |
| :--- | :--- | :--- |
| **Training (Train)** | **Weight Optimization:** The model processes these samples to learn features, adjusting its weights and biases via backpropagation. | **Every Epoch:** Loaded continuously in mini-batches through the `train_loader` during the training loop [205]. |
| **Validation (Val)** | **Hyperparameter Tuning & Overfitting Control:** Used to evaluate the model's performance on unseen data during training [205]. It helps tune hyperparameters (e.g., learning rate, network width) and implement **Early Stopping**. The model *never* updates its weights using these samples [205]. | **End of Every Epoch:** Evaluated once per epoch using the validation DataLoader (`val_loader`) to monitor loss convergence [205, 208]. |
| **Testing (Test)** | **Final Generalization Assessment:** Provides an unbiased estimate of the final model's performance on completely novel data. | **Once at the End:** Used exactly once after all training, hyperparameter tuning, and model selection are completely finalized [205]. |

---

## 2. Why Tuning on the Test Set Must Be Avoided (Data Leakage)

Technically, the **Test Set must remain a secret** to the model and the developer during the training and tuning phases [206]. 

If you use the Test Set to decide when to stop training (e.g., *"I will stop training at Epoch 80 because that is when my test loss is minimized"*), or to select the best hyperparameters, you are committing **data leakage** [206]. Even though the model does not directly run backpropagation on the test samples, your design decisions are guided by them [206]. This "leaks" information from the test set into your model-selection loop, resulting in overfitted models and highly over-optimistic generalization metrics that will fail in real-world deployment.

The **Validation Set** acts as the intermediary proxy for tuning and monitoring, leaving the **Test Set** as the final, pure bench-test of the system [205, 206].

---

## 3. Implementing a 3-Way Split in PyTorch

To align your project with professional best practices, you can easily transition from a simple Train/Test split to a robust Train/Val/Test pipeline [206, 210].

### Step A: Performing the Double Split with scikit-learn
We first split our dataset into a temporary training partition (85%) and a test partition (15%). Then, we split the temporary training partition again to carve out the validation set (15% of the total data) [207].

```python
from sklearn.model_selection import train_test_split

# 1. First split: separate into Train-Temp (85%) and Test (15%)
X_train_temp, X_test, y_train_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# 2. Second split: split Train-Temp into final Train (70% of total) and Val (15% of total)
# 15% is approximately 17.6% of the 85% Train-Temp partition (0.15 / 0.85 ≈ 0.176)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_temp, y_train_temp, test_size=0.176, random_state=42
)

print(f"Total Samples: {len(X)}")
print(f"  - Train Set (70%):      {X_train.shape[0]}")
print(f"  - Validation Set (15%): {X_val.shape[0]}")
print(f"  - Test Set (15%):       {X_test.shape[0]}")
```

### Step B: Constructing Three DataLoaders
Next, wrap each tensor partition into its own `TensorDataset` and create three independent `DataLoader` objects [207, 208]. Only the `train_loader` should have `shuffle=True` [207, 208].

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

# Instantiate DataLoaders
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
```

### Step C: Integrating Validation and Early Stopping into the Training Loop
During training, evaluate the loss on the validation loader at the end of every epoch [205, 208]. Keep track of the lowest validation loss to save the best weights and implement early stopping if the validation loss starts deteriorating [205, 208].

```python
NUM_EPOCHS = 100
best_val_loss = float('inf')
patience = 10
patience_counter = 0

print("Starting training with active validation...")
for epoch in range(1, NUM_EPOCHS + 1):
    # 1. Train on training set
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    
    # 2. Evaluate on validation set (No gradient updates)
    val_loss = evaluate(model, val_loader, criterion) # Uses the same evaluation function
    
    print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # 3. Check for improvement (Early Stopping Logic)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save the best model state
        torch.save(model.state_dict(), "best_predictor_model.pth")
        patience_counter = 0  # Reset counter
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at Epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
            break

print("Training cycle completed.")
```

### Step D: Final Testing Evaluation (Exactly Once)
Once training has stopped, load the best saved model parameters and perform a single final evaluation on the test dataset [205, 209].

```python
# Load the best saved model weights
model.load_state_dict(torch.load("best_predictor_model.pth"))

# Perform final test set evaluation
final_test_loss = evaluate(model, test_loader, criterion)
print(f"\n======================================")
print(f"FINAL EVALUATION ON THE UNSEEN TEST SET")
print(f"======================================")
print(f"Final Test Loss (MSE): {final_test_loss:.4f}")
```

---

## 4. Synthetic vs. Real-World Datasets: When is Validation Critical?

*   **Under Synthetic Data Conditions:**
    When training on structured synthetic data generated through deterministic physical-informed formulas (e.g., semi-empirical photopolymerization kinetics), overfitting is naturally less severe [209]. This is because synthetic data is generally noise-free and strictly follows the mathematical functions used to generate it [209]. While a validation set is still useful, a simple Train/Test split can be acceptable for rapid MVP prototyping [209].
*   **Under Real-World Industrial Conditions:**
    In laboratory, clinical, or industrial environments, data is sparse, noisy, and prone to experimental variance [209]. Real polymerization, viscosity, and curing curves contain complex, non-linear interactions [209]. Under these circumstances, implementing a **Validation Set with Early Stopping is mandatory** [205]. Without it, deep networks will easily memorize the noise present in the training set, leading to poor performance when deployed in production [205, 209].
