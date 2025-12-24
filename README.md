# Workflow-CI - Mohammad Ari Alexander Aziz

## Project: Ethereum Fraud Detection - CI/CD Pipeline

This repository contains **Kriteria 3** (CI/CD Workflow) for the Machine Learning System submission.

---

## 📁 Repository Structure

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── train-model.yml                   # GitHub Actions CI/CD workflow
├── MLProject/
│   ├── modelling.py                          # XGBoost training script
│   ├── conda.yaml                            # Conda environment specification
│   ├── MLProject                             # MLflow Project configuration
│   └── ethereum_fraud_preprocessing.csv     # Preprocessed training data
└── README.md                                 # This file
```

---

## 🎯 Objective

Automate the model training pipeline using **MLflow Projects** and **GitHub Actions**. Every time code is pushed to GitHub, the workflow automatically trains an XGBoost model for fraud detection and saves the results.

---

## 🚀 Quick Start

### Method 1: Let GitHub Actions Run Automatically (Recommended)

**This is the main purpose of this repo - automated CI/CD!**

1. **Push code to GitHub:**
   ```bash
   git push
   ```

2. **GitHub Actions will automatically:**
   - Set up Python environment
   - Install MLflow and dependencies
   - Run the training script
   - Save model artifacts

3. **Check results:**
   - Go to: https://github.com/13222093/Workflow-CI
   - Click "Actions" tab
   - View workflow runs and download artifacts

---

### Method 2: Test Locally with MLflow

**If you want to test locally before pushing to GitHub:**

```bash
# 1. Clone the repo
git clone https://github.com/13222093/Workflow-CI.git
cd Workflow-CI

# 2. Install MLflow
pip install mlflow

# 3. Run the MLflow Project
mlflow run MLProject --experiment-name "Ethereum_Fraud_Detection"

# 4. View results
mlflow ui
```

Then open http://localhost:5000 to see training results.

---

## 🔄 How GitHub Actions CI/CD Works

### Workflow Triggers

The workflow automatically runs when:
- ✅ You push code to `main` branch
- ✅ You create a pull request
- ✅ You manually click "Run workflow" on GitHub

### What Happens During Workflow

1. **Setup:** Install Python 3.12 and Miniconda
2. **Install:** Install MLflow and dependencies
3. **Train:** Run `mlflow run MLProject`
4. **Save:** Upload model artifacts (retained for 90 days)

### How to View Results

1. Go to: https://github.com/13222093/Workflow-CI/actions
2. Click on the latest workflow run
3. Scroll down to "Artifacts" section
4. Download:
   - `model-artifacts` (trained model files)
   - `training-plots` (confusion matrix, feature importance)

---

## ⚙️ MLflow Project Configuration

**File:** `MLProject/MLProject`

Defines the training pipeline with configurable parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 100 | Number of trees in XGBoost |
| `max_depth` | 6 | Maximum depth of each tree |
| `learning_rate` | 0.1 | Learning rate for training |
| `test_size` | 0.2 | Proportion of test data |

**Example: Run with custom parameters**

```bash
mlflow run MLProject \
  -P n_estimators=200 \
  -P max_depth=8 \
  -P learning_rate=0.05
```

---

## 📦 Training Script

**File:** `MLProject/modelling.py`

**What it does:**
1. Loads preprocessed data (`ethereum_fraud_preprocessing.csv`)
2. Splits into train/test sets (80/20)
3. Scales features using PowerTransformer
4. Applies SMOTE for class balancing
5. Trains XGBoost classifier
6. Evaluates performance (accuracy, precision, recall, F1, ROC-AUC)
7. Generates confusion matrix and feature importance plots
8. Logs everything to MLflow

**Output files:**
- `xgboost_fraud_model.pkl` - Trained model
- `scaler_trained.pkl` - Feature scaler
- `confusion_matrix.png` - Confusion matrix visualization
- `feature_importance.png` - Feature importance plot

---

## 📝 Kriteria 3 Level: **Skilled (3 points)**

### Requirements Met:

- ✅ **Basic (2 pts):** MLflow Project + GitHub Actions workflow
- ✅ **Skilled (3 pts):** Artifacts saved via GitHub Actions
- ⬜ **Advanced (4 pts):** Docker image to Docker Hub (optional)

---

## 🧪 Manual Testing

**Test the workflow manually on GitHub:**

1. Go to: https://github.com/13222093/Workflow-CI
2. Click "Actions" tab
3. Select "Train Ethereum Fraud Detection Model"
4. Click "Run workflow" → Select branch `main` → Click "Run workflow"
5. Wait ~5 minutes for completion
6. Download artifacts from the run page

✅ Green checkmark = workflow successful!

---

## 📊 Expected Results

After successful workflow run:

**Metrics (typical):**
- Accuracy: ~95%
- Precision: ~93%
- Recall: ~94%
- F1-Score: ~93%
- ROC-AUC: ~98%

**Artifacts:**
- Model file: ~50 KB
- Scaler file: ~5 KB
- Confusion matrix plot
- Feature importance plot

---

## 👤 Author

**Mohammad Ari Alexander Aziz**
Machine Learning Systems Submission

---

## 🔗 Related Repositories

- **Eksperimen_SML_MohammadAri:** https://github.com/13222093/Eksperimen_SML_MohammadAri (Data preprocessing)

---

## ✅ Submission Checklist

- ✅ GitHub Actions workflow file exists
- ✅ MLflow Project configuration is valid
- ✅ Workflow runs successfully at least once
- ✅ Artifacts are saved and downloadable
- ✅ Repository is public
- ✅ README is clear and informative
