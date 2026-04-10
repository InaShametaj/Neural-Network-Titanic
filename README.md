# Neural-Network-Titanic

# 🚢 Titanic Survival Prediction — Neural Networks

Binary classification project predicting Titanic passenger survival using two neural network architectures built with **scikit-learn**, fully compatible with **JupyterLite / Pyodide** (no PyTorch or TensorFlow required).

---

## 📁 Project Structure

```
titanic-nn/
│
├── Titanic-Dataset.csv          # Raw dataset (UCI / Kaggle)
│
├── titanic_nn_sklearn.py        # Baseline models: BasicNet & DeepNet
├── titanic_analysis.py          # Error analysis, AUC, hyperparameter tuning
│
├── titanic_nn_report.png        # Training curves + architecture summary
├── titanic_error_auc.png        # Confusion matrices, ROC, PR, error breakdowns
├── titanic_hyperparams.png      # Hyperparameter sensitivity sweeps
│
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle Titanic](https://www.kaggle.com/competitions/titanic) |
| Rows | 891 |
| Features (raw) | 12 |
| Features (engineered) | 11 |
| Target | `Survived` (0 = No, 1 = Yes) |
| Class balance | 62% / 38% |

### Preprocessing

- Dropped: `PassengerId`, `Name`, `Ticket`, `Cabin`
- Missing values: `Age` → median, `Embarked` → mode
- Encoding: `Sex` → binary, `Embarked` → one-hot
- Feature engineering:
  - `FamilySize = SibSp + Parch + 1`
  - `IsAlone = (FamilySize == 1)`
  - `FarePerPerson = Fare / FamilySize`
- Scaling: `StandardScaler` (fit on train, applied to test)
- Split: 80% train / 20% test, stratified

---

## 🧠 Models

### BasicNet — 1 Hidden Layer

```
Input (11) → FC(64) → ReLU → Dropout(0.3) → Output(1)
```

| Parameter | Value |
|---|---|
| Hidden layers | 1 |
| Units | 64 |
| Activation | ReLU |
| Regularization | L2 (α = 1e-4) |
| Optimizer | Adam |
| Early stopping | ✓ patience = 20 |
| Parameters | ~770 |

### DeepNet — 5 Hidden Layers

```
Input (11) → FC(256) → FC(128) → FC(64) → FC(32) → FC(16) → Output(1)
```

| Parameter | Value |
|---|---|
| Hidden layers | 5 |
| Units | 256 → 128 → 64 → 32 → 16 |
| Activation | ReLU |
| Regularization | L2 (α = 1e-3) + Adaptive LR |
| Optimizer | Adam |
| Early stopping | ✓ patience = 20 |
| Parameters | ~50,000 |

---

## 📈 Results

| Model | Accuracy | ROC-AUC | Log Loss | Avg Precision |
|---|---|---|---|---|
| BasicNet (baseline) | 0.793 | 0.849 | 0.466 | — |
| DeepNet  (baseline) | 0.805 | **0.865** | **0.436** | — |
| BasicNet (tuned) | 0.760 | 0.838 | 0.487 | — |
| DeepNet  (tuned) | 0.765 | 0.834 | 0.572 | — |

> Baseline models outperform grid-tuned variants on this test set — expected with only ~710 training samples. CV AUC is the more reliable signal at this scale.

### Error Analysis Highlights

- **Pclass 3** has the highest error rate for both models — overlapping survival patterns make it hard to classify
- **Males** are misclassified more often than females (consistent with the "women and children first" signal)
- **False Negatives** (survivors predicted to die) cluster around younger passengers
- DeepNet produces better-calibrated probabilities (lower log loss), even when accuracy is similar

---

## 🔧 Hyperparameter Tuning

Tuned via **5-fold stratified cross-validation** (CV ROC-AUC):

| Hyperparameter | Range tested | Best (BasicNet) | Best (DeepNet) |
|---|---|---|---|
| `alpha` (L2) | 1e-5 → 0.5 | **0.1** | **0.001** |
| `learning_rate_init` | 1e-4 → 5e-2 | **0.005** | **0.0005** |
| `batch_size` | 16, 32, 64, 128, 256 | 32–64 | 32–64 |
| `max_iter` (epochs) | 20 → 500 | ~200 | ~350 |

**Key findings:**
- BasicNet needs stronger regularization (α = 0.1) because it has fewer parameters to naturally resist overfitting
- DeepNet needs a lower learning rate (5e-4) to stabilize gradient flow across 5 layers
- Both models plateau around batch size 32–64; larger batches hurt generalization
- DeepNet benefits from more iterations before convergence

---

## 🖼️ Visualizations

### `titanic_nn_report.png`
Training loss curves, validation accuracy, confusion matrices, and architecture summary table for both baseline models.

### `titanic_error_auc.png`
- Confusion matrices (baseline + tuned, all 4 variants)
- ROC and Precision-Recall curves
- Predicted probability distributions per class
- Error rate broken down by `Pclass`, `Sex`, and `Age`
- Full metrics table (Accuracy, AUC, Log Loss, Brier Score)

### `titanic_hyperparams.png`
Sensitivity sweep charts for all 4 hyperparameters, with best values annotated per model.

---

## 🚀 Getting Started

### Requirements

```
numpy
pandas
scikit-learn
matplotlib
```

No GPU or deep learning framework required. Works in standard Python environments and **JupyterLite** (browser-based).

### Run

```bash
# Install dependencies
pip install numpy pandas scikit-learn matplotlib

# Step 1 — Train baseline models and view report
python titanic_nn_sklearn.py

# Step 2 — Error analysis + hyperparameter sweeps
python titanic_analysis.py
```

Make sure `Titanic-Dataset.csv` is in the same directory.

---

## 💡 Design Decisions

**Why scikit-learn instead of PyTorch/TensorFlow?**
JupyterLite runs Python entirely in the browser via Pyodide. Only packages compiled to WebAssembly are available — PyTorch and TensorFlow are not. `sklearn.neural_network.MLPClassifier` provides full MLP support within that constraint.

**Why two architectures?**
To study the classic bias-variance tradeoff on tabular data. Small datasets often favour simpler models; the comparison makes this concrete rather than assumed.

**Why early stopping on both?**
With 710 training samples, overfitting is the primary risk. Early stopping (patience = 20) acts as a free regularizer without manual epoch tuning.

---

## 📚 References

- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- [sklearn MLPClassifier docs](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [JupyterLite](https://jupyterlite.readthedocs.io/)
