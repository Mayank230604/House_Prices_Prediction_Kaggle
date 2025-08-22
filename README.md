<p align="center">
  <!-- Languages & Core -->
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  
  <!-- ML Libraries -->
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/LightGBM-666666?style=for-the-badge&logo=lightgbm&logoColor=white" />
  
  <!-- Visualization -->
  <img src="https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Seaborn-4C8CBF?style=for-the-badge&logo=plotly&logoColor=white" />
  
  <!-- Tools -->
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" />
</p>

# 🏡 House Prices Prediction — Kaggle Competition

This project tackles the **Kaggle House Prices: Advanced Regression Techniques** competition.
The goal is to predict house prices using various regression models and modern ML workflows.

It demonstrates:

* End-to-end machine learning pipeline
* Feature preprocessing (numeric & categorical)
* Model comparison and cross-validation
* Hyperparameter tuning
* Model persistence & reproducible predictions
* Visual insights into data and results

---

## 📂 Project Structure

```
house_prices_kaggle/
│── data/                  # train.csv, test.csv, sample_submission.csv
│── models/                # Saved trained pipelines + metadata
│── outputs/               # Submission CSVs for Kaggle
│── figures/               # EDA and feature importance plots
│── src/
│   └── train_house_prices.py   # Main training & evaluation script
│── requirements.txt
│── README.md
│── .gitignore
```

---

## ⚙️ Workflow

### 1️⃣ Data Loading & Preprocessing

* Handle missing values:
  * Numeric → median
  * Categorical → most frequent
* Scale numeric features
* One-hot encode categorical features
* Log-transform target `SalePrice` for stability

### 2️⃣ Model Training

* **Ridge Regression**
* **Lasso Regression**
* **Random Forest Regressor**
* **HistGradientBoosting Regressor**

Each model is evaluated with **5-Fold Cross-Validation** using **RMSE** as the metric.

### 3️⃣ Hyperparameter Tuning

* Best base model (typically **HistGBR**) is selected
* Tuned using **RandomizedSearchCV** to improve performance

### 4️⃣ Evaluation & Outputs

* Cross-validation scores (mean ± std)
* Best hyperparameters logged in `models/` metadata
* Feature importance plots (if supported by model)
* Test predictions saved to `outputs/` (submission-ready CSV)

---

## 📊 Results

**Cross-validation RMSE (5-fold):**

| Model                | CV RMSE (mean ± std) |
| -------------------- | -------------------- |
| Ridge                | 0.1485 ± 0.0407      |
| Lasso                | 0.1443 ± 0.0424      |
| Random Forest        | 0.1452 ± 0.0191      |
| HistGBR              | 0.1355 ± 0.0170      |

✅ **Best base model:** HistGradientBoostingRegression

**Hyperparameter Tuning (RandomizedSearchCV):**

```python
Best params:
{
    'model__min_samples_leaf': 20,
    'model__max_leaf_nodes': 15,
    'model__max_depth': 8,
    'model__learning_rate': 0.1,
    'model__l2_regularization': 0.1
}

Best CV RMSE: 0.1319
```

---

## 📈 Results & Figures

Figures are saved in the `figures/` directory:

* Correlation heatmap of key features
* Distribution plots of SalePrice (log-transformed vs original)
* Feature importance (top 20 predictors)
* CV performance comparison across models

These visualizations provide insights into **feature relevance**, **data quality**, and **model behavior**, making the workflow more transparent.

---

## 🚀 Getting Started

1. Clone this repo:

```bash
git clone https://github.com/Mayank230604/House_Prices_Prediction_Kaggle.git
cd House_Prices_Prediction_Kaggle
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place Kaggle competition data inside `data/`:

* `train.csv`
* `test.csv`
* `sample_submission.csv`

4. Run the training script:

```bash
python src/train_house_prices.py
```

5. Check results:

* `models/` → Saved best pipeline + metadata
* `outputs/` → Submission CSV for Kaggle
* `figures/` → Plots

---

## 📌 Key Insights

* **HistGradientBoostingRegressor** consistently outperformed other models
* Hyperparameter tuning improved RMSE further
* Scaling + one-hot encoding were crucial for linear models
* Feature importance plots highlight key drivers of house prices

---

## 🏆 Kaggle Competition Link

[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)


