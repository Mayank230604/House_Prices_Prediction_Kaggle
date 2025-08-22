# 🏡 House Prices Prediction — Kaggle Competition

This project tackles the **Kaggle House Prices: Advanced Regression Techniques** competition.  
The goal is to predict house prices using various regression models and modern ML workflows.  

It demonstrates:
- Complete end-to-end ML pipeline
- Feature preprocessing (numeric & categorical)
- Model comparison and cross-validation
- Hyperparameter tuning
- Model persistence + reproducible predictions
- Visual insights into data & results

---

## 📂 Project Structure

```

house\_prices\_kaggle/
│── data/                  # train.csv, test.csv, sample\_submission.csv
│── models/                # saved trained pipelines + metadata
│── outputs/               # submission CSVs for Kaggle
│── figures/               # EDA and feature importance plots
│── src/
│   └── train\_house\_prices.py   # main training & evaluation script
│── requirements.txt
│── README.md
│── .gitignore

```

---

## ⚙️ Workflow

1. **Data Loading & Preprocessing**
   - Handle missing values (numeric → median, categorical → most frequent)
   - Scale numeric features
   - One-hot encode categorical features
   - Log-transform target `SalePrice` for stability

2. **Model Training**
   - Ridge Regression  
   - Lasso Regression  
   - Random Forest Regressor  
   - HistGradientBoosting Regressor  

   Each model is cross-validated using **5-Fold CV** with RMSE as the metric.

3. **Hyperparameter Tuning**
   - The best base model is selected (typically **HistGBR**)  
   - Tuned via **RandomizedSearchCV** for improved performance  

4. **Evaluation & Outputs**
   - Cross-validation scores (mean ± std)  
   - Best hyperparameters logged in `models/` metadata  
   - Feature importance plots (if supported by model)  
   - Test predictions saved to `outputs/` (submission-ready CSV)  

---

## 📊 Example Results

Cross-validation RMSE (5-fold):

```

```
   Ridge | CV RMSE: 0.1485 (+/- 0.0407)
   Lasso | CV RMSE: 0.1443 (+/- 0.0424)
```

RandomForest | CV RMSE: 0.1452 (+/- 0.0191)
HistGBR | CV RMSE: 0.1355 (+/- 0.0170)

```

Best Model after tuning:
```

HistGBR with params:
{
'model\_\_learning\_rate': 0.1,
'model\_\_max\_depth': 8,
'model\_\_max\_leaf\_nodes': 15,
'model\_\_min\_samples\_leaf': 20,
'model\_\_l2\_regularization': 0.1
}

````

---

## 🚀 Getting Started

1. Clone this repo:
   ```bash
   git clone https://github.com/Mayank230604/House_Prices_Prediction_Kaggle.git
   cd House_Prices_Prediction_Kaggle
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Place Kaggle competition data inside `data/`:

   * `train.csv`
   * `test.csv`
   * `sample_submission.csv`

4. Run training script:

   ```bash
   python src/train_house_prices.py
   ```

5. Check results in:

   * `models/` → saved best pipeline + metadata
   * `outputs/` → submission CSV for Kaggle
   * `figures/` → plots

---

## 📌 Key Insights

* **HistGradientBoostingRegressor** consistently outperformed other models.
* Hyperparameter tuning improved RMSE further.
* Scaling + one-hot encoding were crucial for linear models.
* Feature importance plots highlight key drivers of house prices.

---

## 🏆 Kaggle Competition Link

[House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

---

