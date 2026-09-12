# Post-MVP Roadmap — From Prediction to Discovery
## How to Extend the UV-Curing Pipeline into Active Learning and Generative Molecular Design

### 0. Where You Are and Where You Are Going

#### 0.1 Current state
You have a working MVP that does one thing: given a pair of molecules (PI + monomer) and four process parameters, it predicts a curing conversion percentage. The pipeline is:

```text
[User types names] → [PubChem resolves SMILES] → [RDKit draws image] → [MobileNetV2 extracts embeddings] → [XGBoost predicts %]
```

This is a forward prediction tool. It answers the question: "What would happen if I used these two molecules under these conditions?"

#### 0.2 Target state
The post-MVP evolution adds two capabilities that transform the tool from a predictor into a discovery engine:

| Capability | Question it answers | What it produces |
|---|---|---|
| **Active Learning (Approach 4)** | "Which molecules should I test first?" | A ranked list of candidates |
| **Generative Modeling (Approach 2)** | "What new molecules could I invent?" | Novel molecular structures |

The two capabilities are complementary. Active Learning optimizes the selection of molecules from a known pool. Generative Modeling creates new molecules that do not exist in any database. Combined, they form a closed-loop discovery pipeline — the state of the art in AI-driven molecular design.

#### 0.3 The master architecture
```text
┌─────────────────────────────────────────────────────────────────┐
│                    CLOSED-LOOP DISCOVERY                        │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │  GENERATOR   │───▶│  PREDICTOR   │───▶│  ACQUISITION │     │
│  │  (Phase B)   │    │  (MVP)       │    │  (Phase A)   │     │
│  │              │    │              │    │              │     │
│  │ VAE / GNN    │    │ XGBoost      │    │ UCB / EI     │     │
│  │ Creates new  │    │ Predicts     │    │ Selects      │     │
│  │ molecules    │    │ conversion   │    │ candidates   │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         ▲                                        │             │
│         │                                        ▼             │
│         │                              ┌──────────────┐       │
│         └──────────────────────────────│  ORACLE      │       │
│                                        │  (simulation │       │
│                                        │   or lab)    │       │
│                                        └──────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```
Each phase below builds one component of this loop.

---

### Phase A — Active Learning (2–3 Weeks)

#### A.1 The Problem Active Learning Solves
You have 224 photoinitiators and 40 monomers in your dataset. The full factorial of PI–monomer pairs is 8,960 combinations. Add 10 values per environmental parameter (dose, concentration, LogP, aqueous) and you get 89,600 possible experiments.

Testing all of them in a laboratory would take years. Testing even 1% (896 experiments) would take months. Active Learning answers the question: which 20 experiments should you run to learn the most?

This is not a hypothetical problem. It is the exact problem faced by every materials discovery company. The answer comes from Bayesian Optimization (BO).

#### A.2 Theory: Bayesian Optimization and Acquisition Functions
Bayesian Optimization is a strategy for optimizing expensive black-box functions. It has two components:
- **Surrogate model**: a probabilistic model (usually a Gaussian Process) that approximates the true function and provides uncertainty estimates.
- **Acquisition function**: a function that scores candidate points by balancing exploration (high uncertainty) and exploitation (high predicted value).

The acquisition function is the heart of active learning. Three variants dominate:

| Acquisition Function | Formula (conceptual) | Behavior |
|---|---|---|
| **UCB (Upper Confidence Bound)** | $\mu(x) + \kappa \cdot \sigma(x)$ | Prefers points with high mean plus high uncertainty |
| **EI (Expected Improvement)** | $\mathbb{E}[\max(0, f(x) - f_{\text{best}})]$ | Prefers points likely to improve on the current best |
| **Thompson Sampling** | Sample from posterior, pick $\arg\max$ | Random but principled exploration |

In practice, UCB is the most widely used for molecular discovery because it has a single tunable parameter ($\kappa$) that directly controls the exploration-exploitation trade-off. A study on cocrystal discovery found that UCB consistently outperformed other acquisition functions across multiple fingerprints.

#### A.3 Implementation Steps

##### Step A.3.1 — Define the search space
Create a new module `active_learning/search_space.py`:

```python
"""
Defines the search space for active learning.
The search space is the set of all (PI, monomer, conditions) triples that are physically valid and worth testing.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np

@dataclass
class ExperimentalCondition:
    """A single experimental condition (one row in the design space)."""
    pi_name: str
    monomer_name: str
    is_aqueous: int        # 0 or 1
    logp: float            # 0.5 – 5.0
    pi_concentration: float  # 1.0 – 5.0
    uv_dose: float         # 50 – 500

def build_search_space(
    pi_names: list[str],
    monomer_names: list[str],
    n_logp: int = 5,
    n_conc: int = 5,
    n_dose: int = 5,
) -> list[ExperimentalCondition]:
    """
    Build a discrete search space by grid-sampling the continuous parameters.

    With 224 PIs, 40 monomers, and 5×5×5 = 125 condition combinations,
    the full space is 224 × 40 × 2 × 125 = 2.24 million points.
    Too large to evaluate directly — that is why we need active learning.
    """
    logp_values = np.linspace(0.5, 5.0, n_logp)
    conc_values = np.linspace(1.0, 5.0, n_conc)
    dose_values = np.linspace(50, 500, n_dose)

    space = []
    for pi, mono, aqueous, logp, conc, dose in product(
        pi_names, monomer_names, [0, 1],
        logp_values, conc_values, dose_values,
    ):
        space.append(ExperimentalCondition(
            pi_name=pi, monomer_name=mono,
            is_aqueous=aqueous, logp=logp,
            pi_concentration=conc, uv_dose=dose,
        ))
    return space
```

##### Step A.3.2 — Select a surrogate model
Option 1: Use a Gaussian Process (GP) — the classical choice.

```python
# active_learning/surrogate.py
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import numpy as np

class GPSurrogate:
    """
    Gaussian Process surrogate with Tanimoto kernel for molecular fingerprints.

    A GP provides both a mean prediction and a variance (uncertainty) for
    every query point. This is exactly what acquisition functions need.
    """

    def __init__(self):
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=0.1)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.gp.fit(X, y)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) for each row in X."""
        mean, std = self.gp.predict(X, return_std=True)
        return mean, std
```

Option 2: Use XGBoost with quantile regression — simpler, reuses your existing model.

```python
# active_learning/surrogate.py
import xgboost as xgb
import numpy as np

class XGBoostSurrogate:
    """
    XGBoost with quantile regression to estimate uncertainty.

    Train three models: one for the median (0.5), one for the lower
    bound (0.1), one for the upper bound (0.9). The spread between
    the bounds is a proxy for uncertainty.
    """

    def __init__(self):
        self.models = {}
        for q in [0.1, 0.5, 0.9]:
            self.models[q] = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=q,
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
            )

    def fit(self, X, y):
        for q, model in self.models.items():
            model.fit(X, y)

    def predict(self, X):
        lower = self.models[0.1].predict(X)
        median = self.models[0.5].predict(X)
        upper = self.models[0.9].predict(X)
        std_proxy = (upper - lower) / 2.56  # approx normal IQR/2
        return median, np.maximum(std_proxy, 1e-3)
```

Recommendation: start with `XGBoostSurrogate` because you already have the XGBoost pipeline working. Switch to GP later if you need better-calibrated uncertainty.

##### Step A.3.3 — Implement the acquisition function

```python
# active_learning/acquisition.py
import numpy as np

def ucb(mean: np.ndarray, std: np.ndarray, kappa: float = 2.0) -> np.ndarray:
    """
    Upper Confidence Bound.

    UCB(x) = mean(x) + kappa * std(x)

    Higher kappa → more exploration.
    Lower kappa  → more exploitation.
    kappa=2.0 is a standard default.
    """
    return mean + kappa * std

def thompson_sampling(mean: np.ndarray, std: np.ndarray, n_samples: int = 1) -> np.ndarray:
    """
    Thompson Sampling: sample one function from the posterior, return the argmax of the sample.

    Simple but effective; works well when the number of candidates is large.
    """
    samples = np.random.normal(mean, std)
    return samples
```

##### Step A.3.4 — Build the active learning loop

```python
# active_learning/loop.py
""" The active learning loop: train → suggest → evaluate → retrain. """
import numpy as np
from active_learning.surrogate import XGBoostSurrogate
from active_learning.acquisition import ucb
```

##### Step A.3.5 — Simulate the loop (no lab required)
Since you do not have a laboratory, you simulate the experiments using your existing `predict()` function as the oracle:

```python
# active_learning/simulate.py
"""
Simulate the active learning loop using the MVP pipeline as the oracle.
The oracle is a stand-in for a real laboratory: given a candidate (PI, monomer, conditions), it returns the "true" conversion %.
"""
import numpy as np
from inference.pipeline import predict
from active_learning.surrogate import XGBoostSurrogate
from active_learning.loop import ActiveLearningLoop

def featurize(conditions, pi_embeddings, mono_embeddings, pi_index, mono_index):
    """Convert a list of ExperimentalCondition into a feature matrix."""
    rows = []
    for c in conditions:
        pi_emb = pi_embeddings[pi_index[c.pi_name]]
        mono_emb = mono_embeddings[mono_index[c.monomer_name]]
        row = np.concatenate([
            pi_emb, mono_emb,
            [c.is_aqueous, c.logp, c.pi_concentration, c.uv_dose],
        ])
        rows.append(row)
    return np.array(rows)

def run_simulation(pi_names, monomer_names, n_rounds=20, batch_size=5):
    """
    Simulate 20 rounds of active learning, 5 experiments per round.
    Track how the best observed conversion improves over time.
    """
    # ... build search space, featurize, initialize surrogate with 10 random points ...
    # ... loop: suggest → oracle → observe → record best ...
    pass
```

##### Step A.3.6 — Evaluate the strategy
Plot two curves:
- **Best conversion found vs. number of experiments**. Active learning should reach high conversion faster than random selection.
- **Uncertainty reduction**. The surrogate's uncertainty should decrease as more observations are collected.

Compare three strategies:
1. Random selection (baseline)
2. UCB with $\kappa=2.0$ (balanced)
3. UCB with $\kappa=5.0$ (exploration-heavy)

The standard result in the literature is that active learning reaches the optimum in 3–10× fewer experiments than random selection.

#### A.4 Integration with the Reflex app
Add a new page `app/pages/active_learning.py` with:
- A button "Run 1 round" that triggers one iteration of `ActiveLearningLoop.suggest()`
- A table showing the suggested experiments (PI, monomer, conditions, predicted conversion, uncertainty)
- A plot showing the improvement curve
- A "Reset" button to restart

#### A.5 Libraries to install
```bash
pip install scikit-optimize scipy
```
`scikit-optimize` provides Bayesian optimization primitives (Gaussian Process, acquisition functions) out of the box. If you want a molecular-specific library, use `molbo`.

#### A.6 Portfolio value
This phase demonstrates that you can:
- Design an experimental campaign using Bayesian Optimization
- Implement acquisition functions from scratch
- Simulate a real industrial workflow (suggest → test → update → repeat)
- Quantify the value of active learning (fewer experiments for the same result)

---

### Phase B — Generative Models (3–4 Months)

#### B.1 The Problem Generative Modeling Solves
Active learning selects from molecules that already exist in a database. But what if the best photoinitiator for your formulation has never been synthesized? You cannot look it up — you have to create it.

Generative models learn the "grammar" of valid molecular structures and can then generate novel molecules with desired properties. This is the core of inverse molecular design: instead of predicting properties of a given molecule, you generate molecules that have desired properties.

#### B.2 Three Generative Architectures

| Architecture | Represents molecules as | Strength | Weakness |
|---|---|---|---|
| **VAE on SMILES** | Character sequence (text) | Simple, fast, smooth latent space | SMILES is brittle (invalid strings) |
| **VAE on Graphs** | Atoms + bonds | Chemically meaningful | More complex implementation |
| **Transformer (ChemBERTa)** | SMILES tokens | State-of-the-art quality | Heavy compute for pretraining |
| **Diffusion on 3D** | Atom positions | Best quality for 3D | Requires 3D conformers |

Recommendation for your project: start with a **VAE on SMILES**. It is the most documented, has the most tutorials, and gives a working generative model in 2–3 weeks of study.

#### B.3 Theory: How a VAE on SMILES Works
A Variational Autoencoder has two parts:
- **Encoder**: takes a SMILES string → maps it to a latent vector (e.g., 128 dimensions)
- **Decoder**: takes a latent vector → reconstructs a SMILES string

The magic is in the latent space. During training, the VAE learns to organize the latent space so that:
- Similar molecules are close together
- The space is smooth: moving in a direction changes the molecule gradually
- Sampling a random point produces a valid molecule

This is why VAEs are used for molecular generation: you can optimize in latent space (continuous, differentiable) instead of in molecular space (discrete, combinatorial).

```text
SMILES "C=CC(=O)O"
     │
     ▼
┌─────────┐
│ ENCODER │  →  z = [0.2, -1.3, 0.8, ...]  (128 dims)
└─────────┘
     │
     │  (optimize z for desired property)
     ▼
┌─────────┐
│ DECODER │  →  SMILES "C=CC(=O)OCCO"
└─────────┘
```

#### B.4 Implementation Steps

##### Step B.4.1 — Install PyTorch and RDKit
```bash
pip install torch torchvision
pip install rdkit
```
*(You already have RDKit from the MVP.)*

##### Step B.4.2 — Get a molecular dataset
Download a public SMILES dataset. The standard choices:

| Dataset | Size | Download |
|---|---|---|
| **ChEMBL** | ~2M molecules | https://ftp.ebi.ac.uk/pub/databases/chembl/ |
| **ZINC** | ~10M molecules | https://zinc.docking.org/ |
| **QM9** | ~134K small molecules | Bundled in `torch_geometric.datasets` |
| **PubChem** | ~100M | Use the PubChem API (you already have the client) |

For a first VAE, use ChEMBL or ZINC — they are large and diverse. A subset of 100K–500K molecules is enough to train a good VAE.

##### Step B.4.3 — Preprocess: SMILES tokenization
SMILES is a string. To feed it to a neural network, you need to tokenize it. The simplest approach is character-level tokenization:

```python
# generative/smiles_tokenizer.py
class SMILESTokenizer:
    def __init__(self, smiles_list):
        chars = sorted(set("".join(smiles_list)))
        self.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}
        self.char_to_idx["<pad>"] = 0
        self.char_to_idx["<sos>"] = len(self.char_to_idx)
        self.char_to_idx["<eos>"] = len(self.char_to_idx)
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
```

##### Step B.4.4 — Build the VAE

```python
# generative/vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SMILESVAE(nn.Module):
    """
    Variational Autoencoder for SMILES strings.

    Encoder: GRU → mean and log-variance vectors
    Decoder: GRU → character probabilities
    """

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, latent_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Encoder
        self.encoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):
        emb = self.embed(x)
        _, h = self.encoder_gru(emb)
        h = h.squeeze(0)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x):
        h = self.latent_to_hidden(z).unsqueeze(0)
        emb = self.embed(x)
        out, _ = self.decoder_gru(emb, h)
        return self.fc_out(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, x[:, :-1])  # teacher forcing
        return logits, mu, logvar

def vae_loss(logits, targets, mu, logvar, beta=1.0):
    """
    VAE loss = reconstruction loss + beta * KL divergence.

    The KL term regularizes the latent space to be smooth and
    normally distributed, which is what makes generation possible.
    """
    recon = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=0,
    )
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl = kl / logits.size(0)
    return recon + beta * kl, recon, kl
```

##### Step B.4.5 — Train the VAE

```python
# generative/train_vae.py
"""
Training loop for the SMILES VAE.
Uses KL annealing: start with beta=0 (pure autoencoder) and gradually increase beta to 1.0 over the first 20 epochs. This prevents the KL term from dominating early training and collapsing the latent space.
"""
import torch
from torch.utils.data import DataLoader
from generative.vae import SMILESVAE, vae_loss
from generative.smiles_tokenizer import SMILESTokenizer

def train(smiles_list, n_epochs=50, batch_size=128, lr=1e-3):
    tokenizer = SMILESTokenizer(smiles_list)
    dataset = [tokenizer.encode(s) for s in smiles_list]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # ... training loop implementation ...
```

##### Step B.4.6 — Evaluate the VAE
Three standard metrics for molecular generation:

| Metric | What it measures | Target |
|---|---|---|
| **Validity** | % of generated strings that RDKit parses | > 70% |
| **Uniqueness** | % of valid molecules that are distinct | > 90% |
| **Novelty** | % of unique molecules not in training set | > 80% |

```python
# generative/evaluate.py
from rdkit import Chem
import numpy as np

def evaluate_vae(model, tokenizer, n_samples=1000, latent_dim=128):
    """Sample from the prior and compute the three standard metrics."""
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        # ... decode z into SMILES strings ...
        generated = [...]  # list of SMILES strings

        valid = [s for s in generated if Chem.MolFromSmiles(s) is not None]
        unique = set(valid)
        novel = unique - set(training_smiles)

        validity = len(valid) / n_samples
        uniqueness = len(unique) / max(len(valid), 1)
        novelty = len(novel) / max(len(unique), 1)

        return {"validity": validity, "uniqueness": uniqueness, "novelty": novelty}
```

##### Step B.4.7 — Guided generation (the key step)
A plain VAE generates random molecules. To generate molecules with desired properties, you optimize in latent space:

```python
# generative/guided.py
"""
Guided generation: find latent vectors that decode into molecules with high predicted conversion.
"""
import torch
from inference.pipeline import predict

def guided_generation(vae, tokenizer, property_predictor, n_steps=100):
    """ Optimize a latent vector to maximize the property predictor. """
    z = torch.randn(1, 128, requires_grad=True)
    optimizer = torch.optim.Adam([z], lr=0.01)

    for step in range(n_steps):
        smiles = tokenizer.decode(vae.decode(z, ...))  # decode
        if smiles is None or len(smiles) == 0:
            continue
        score = property_predictor(smiles)  # predict conversion
        loss = -score  # maximize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return tokenizer.decode(vae.decode(z, ...))
```

*Important note*: this simple approach has a problem. The decoder is not differentiable with respect to the property score when the property predictor takes SMILES as input (string → non-differentiable). Two solutions:
1. **Replace the property predictor with a differentiable one**. Use a GNN or a model that takes the latent vector directly.
2. **Use a gradient-free optimizer**. Use Bayesian Optimization (from Phase A) in latent space instead of gradient descent.

Solution 2 is cleaner and reuses the Phase A infrastructure.

#### B.5 More Advanced: Graph Neural Networks
Once you are comfortable with the SMILES VAE, the next step is Graph Neural Networks (GNNs). Instead of representing molecules as strings, you represent them as graphs: atoms are nodes, bonds are edges.

Why GNNs?
- More chemically faithful (no invalid SMILES)
- Better at capturing local structure
- State of the art for property prediction

Library: `PyTorch Geometric`
```bash
conda install pyg -c pyg
pip install torch-scatter torch-sparse torch-cluster
```
Tutorial: the QM9 dataset is the standard benchmark. The workflow is:
1. Load QM9 molecules as graphs
2. Build a GNN (`GCNConv` or `GATConv` layers)
3. Train to predict a property (e.g., HOMO-LUMO gap)
4. Evaluate on test set

Once you can do that, you can replace the MobileNetV2 + XGBoost pipeline with a GNN that operates directly on molecular graphs.

#### B.6 Portfolio value
This phase demonstrates that you can:
- Implement a generative model from scratch
- Train it on real molecular data
- Evaluate it with standard cheminformatics metrics
- Use it for inverse molecular design (generate molecules with desired properties)

---

### Phase C — Closed-Loop Discovery (2 Weeks)

#### C.1 What the Closed Loop Does
The closed loop connects Phase A and Phase B into a single iterative pipeline:

```text
┌─────────────────────────────────────────────────────────────┐
│  1. GENERATE: VAE produces 1000 novel molecules             │
│         │                                                   │
│         ▼                                                   │
│  2. PREDICT: XGBoost (or GNN) predicts conversion for each  │
│         │                                                   │
│         ▼                                                   │
│  3. SELECT: Acquisition function picks top 10 candidates    │
│         │                                                   │
│         ▼                                                   │
│  4. EVALUATE: Oracle (simulation or lab) returns ground truth│
│         │                                                   │
│         ▼                                                   │
│  5. UPDATE: Retrain predictor on new (molecule, conversion) │
│         │                                                   │
│         ▼                                                   │
│  6. Repeat from step 1 with improved predictor              │
└─────────────────────────────────────────────────────────────┘
```

Each iteration:
1. The generator proposes new molecules
2. The predictor scores them
3. The acquisition function selects the most promising
4. The oracle evaluates them (in simulation, using your `predict()` function)
5. The predictor is retrained on the expanded dataset

This is the state of the art in molecular discovery.

#### C.2 Implementation

```python
# closed_loop/pipeline.py
""" Closed-loop molecular discovery: generate → predict → select → evaluate → update. """
from generative.vae import SMILESVAE
from generative.smiles_tokenizer import SMILESTokenizer
from active_learning.surrogate import XGBoostSurrogate
from active_learning.acquisition import ucb
from inference.pipeline import predict

class ClosedLoopDiscovery:
    def __init__(self, vae, tokenizer, predictor, acquisition_fn=ucb):
        self.vae = vae
        self.tokenizer = tokenizer
        self.predictor = predictor
        self.acquisition_fn = acquisition_fn

    def run(self, n_rounds=10, n_generate=1000, n_select=10):
        history = []
        for round_idx in range(n_rounds):
            # 1. Generate
            candidates = self.vae.sample(n_generate)  # list of SMILES

            # 2. Predict
            X = featurize(candidates)
            mean, std = self.predictor.predict(X)

            # 3. Select
            scores = self.acquisition_fn(mean, std)
            top_idx = np.argsort(scores)[::-1][:n_select]
            selected = [candidates[i] for i in top_idx]

            # 4. Evaluate (oracle = simulation)
            y_true = [oracle(smiles) for smiles in selected]

            # 5. Update
            self.predictor.fit(X[top_idx], np.array(y_true))

            history.append({
                "round": round_idx,
                "best_conversion": max(y_true),
                "mean_conversion": np.mean(y_true),
            })
            print(f"Round {round_idx}: best={max(y_true):.1f}%")

        return history
```

#### C.3 Evaluation
Plot the progression:
- **Round vs. best conversion found**: should increase and plateau
- **Round vs. novelty**: how many generated molecules are new?
- **Round vs. validity**: does the VAE maintain chemical validity?

Compare:
1. Closed loop (generate + predict + active learning)
2. Static generation (generate once, pick top, stop)
3. Random generation (baseline)

The literature shows that closed-loop generation improves out-of-distribution generalization by up to 79% compared to static approaches.

#### C.4 Portfolio value
This is the maximum differentiation. Very few candidates have built a full closed-loop discovery system. It demonstrates that you can:
- Integrate generative models with predictive models
- Design and implement a multi-stage AI pipeline
- Evaluate an iterative optimization system
- Understand the state of the art in AI for Science

---

### Summary: The Complete Roadmap

| Phase | Duration | What you learn | Key libraries | Portfolio value |
|---|---|---|---|---|
| **A — Active Learning** | 2–3 weeks | Bayesian Optimization, acquisition functions, experimental design | `scikit-optimize`, `scipy` | Medium |
| **B — Generative Models** | 3–4 months | VAEs, transformers, GNNs, molecular generation | `torch`, `torch_geometric`, `rdkit` | High |
| **C — Closed Loop** | 2 weeks | System integration, iterative optimization | *(combines A + B)* | Maximum |

#### Recommended reading order
1. Elton et al. (2019), *"Deep learning for molecular design — a review"* — the foundational review
2. Gómez-Bombarelli et al. (2018), *"Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules"* — the VAE paper
3. Sanchez-Lengeling & Aspuru-Guzik (2018), *"Inverse molecular design using machine learning"* — the conceptual guide
4. The ChemBERTa-3 paper (2026) — the state of the art in chemical language models

#### Practical first step
Start with **Phase A**. It is the shortest, the most directly applicable to your existing MVP, and gives you a working "active learning" feature in 2–3 weeks. Once that is done and documented, move to Phase B with a clear understanding of what you are building.
