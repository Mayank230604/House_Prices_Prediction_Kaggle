import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

RANDOM_STATE = 42

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    # --- Paths
    DATA_DIR = "data"
    MODELS_DIR = "models"
    OUTPUTS_DIR = "outputs"
    FIGURES_DIR = "figures"
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    # --- Load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Separate target and features
    y = np.log1p(train["SalePrice"].values)  # log-transform target
    X = train.drop(columns=["SalePrice"])
    test_ids = test["Id"].copy()

    # --- Preprocessing
    numeric_selector = selector(dtype_include=np.number)
    categorical_selector = selector(dtype_include=object)

    numeric_preprocess = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_preprocess = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_preprocess, numeric_selector),
        ("cat", categorical_preprocess, categorical_selector),
    ])

    # --- Candidate models
    candidates = {
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.001, max_iter=10000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "HistGBR": HistGradientBoostingRegressor(
            learning_rate=0.05, max_depth=None, random_state=RANDOM_STATE
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = "neg_root_mean_squared_error"

    print("🔎 Cross-validating models (metric: RMSE)...")
    cv_results = []
    for name, model in candidates.items():
        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        mean_rmse = -scores.mean()
        std_rmse = scores.std()
        cv_results.append((name, mean_rmse, std_rmse))
        print(f"{name:>12} | CV RMSE: {mean_rmse:.4f} (+/- {std_rmse:.4f})")

    # Pick the best
    cv_results.sort(key=lambda t: t[1])
    best_name = cv_results[0][0]
    print(f"\n✅ Best base model by CV: {best_name}")

    # --- Hyperparameter search
    if best_name == "HistGBR":
        base_model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
        param_dist = {
            "model__learning_rate": [0.03, 0.05, 0.07, 0.1],
            "model__max_depth": [None, 4, 6, 8],
            "model__max_leaf_nodes": [15, 31, 63],
            "model__min_samples_leaf": [5, 10, 20],
            "model__l2_regularization": [0.0, 0.1, 0.5, 1.0],
        }
    elif best_name == "RandomForest":
        base_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        param_dist = {
            "model__n_estimators": [300, 500, 800],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["auto", "sqrt", 0.5],
        }
    elif best_name == "Ridge":
        base_model = Ridge(random_state=RANDOM_STATE)
        param_dist = {"model__alpha": np.logspace(-3, 3, 20)}
    else:  # Lasso
        base_model = Lasso(random_state=RANDOM_STATE, max_iter=10000)
        param_dist = {"model__alpha": np.logspace(-4, 0, 20)}

    base_pipe = Pipeline(steps=[("prep", preprocessor), ("model", base_model)])
    search = RandomizedSearchCV(
        estimator=base_pipe,
        param_distributions=param_dist,
        n_iter=25,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )

    print("\n🛠  Hyperparameter search (RandomizedSearchCV)...")
    search.fit(X, y)
    best_pipe = search.best_estimator_
    best_cv_rmse = -search.best_score_
    print(f"🔧 Best params: {search.best_params_}")
    print(f"🏁 Best CV RMSE: {best_cv_rmse:.4f}")

    # --- Fit best on all training data & predict
    print("\n📦 Fitting best model on full training data...")
    best_pipe.fit(X, y)

    print("🧮 Generating test predictions...")
    test_preds_log = best_pipe.predict(test)
    test_preds = np.expm1(test_preds_log)

    # --- Save submission
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_path = os.path.join(OUTPUTS_DIR, f"submission_{best_name}_{ts}.csv")
    pd.DataFrame({"Id": test_ids, "SalePrice": test_preds}).to_csv(sub_path, index=False)
    print(f"📝 Submission saved: {sub_path}")

    # --- Save model + metadata
    model_path = os.path.join(MODELS_DIR, f"best_pipeline_{best_name}.pkl")
    joblib.dump(best_pipe, model_path)
    print(f"💾 Trained pipeline saved: {model_path}")

    meta = {
        "best_model": best_name,
        "best_params": search.best_params_,
        "cv_rmse": round(best_cv_rmse, 5),
        "created_at": ts,
    }
    with open(os.path.join(MODELS_DIR, f"best_pipeline_{best_name}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("🧷 Metadata saved.")

    # --- 📊 Save Figures
    try:
        y_train_pred = best_pipe.predict(X)
        residuals = y - y_train_pred

        # Residuals plot
        plt.figure(figsize=(8,6))
        plt.scatter(y_train_pred, residuals, alpha=0.5)
        plt.axhline(0, color="red", linestyle="--")
        plt.xlabel("Predicted SalePrice (log)")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predictions")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "residuals_vs_predictions.png"))
        plt.close()

        # Distribution plot
        plt.figure(figsize=(8,6))
        sns.kdeplot(y, label="Actual", fill=True)
        sns.kdeplot(y_train_pred, label="Predicted", fill=True)
        plt.title("Distribution of Actual vs Predicted SalePrice (log)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "actual_vs_predicted.png"))
        plt.close()

        # Feature importances if available
        model = best_pipe.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            feat_names = best_pipe.named_steps["prep"].get_feature_names_out()
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)[:20]
            plt.figure(figsize=(10,6))
            feat_imp.plot(kind="barh")
            plt.gca().invert_yaxis()
            plt.title("Top 20 Feature Importances")
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_DIR, "feature_importances.png"))
            plt.close()

        print("📊 Figures saved in 'figures/' folder.")

    except Exception as e:
        print("⚠️ Could not generate figures:", e)

if __name__ == "__main__":
    main()
