"""
How to Run
----
python3 ml_analysis.py --screen data/screentime.csv --spend data/spending.csv
"""

import argparse, os, sys, json, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.feature_selection import RFE

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

ROOT_OUT = Path("ml_analysis_output")
SUBDIRS = [
    "0_raw_inspection",
    "1_time_series",
    "2_modeling",
]
for sd in SUBDIRS:
    (ROOT_OUT / sd).mkdir(parents=True, exist_ok=True)

EXPLAIN_DIR = ROOT_OUT / "3_explainability"

META_PATH = ROOT_OUT / "model_meta.json"

def load_and_enrich_data(screen_path: Path, spend_path: Path) -> pd.DataFrame:
    screen = pd.read_csv(screen_path, sep=";")
    if "Date" not in screen.columns:
        raise ValueError("Missing 'Date' column in screentime CSV")
    screen["Date"] = pd.to_datetime(screen["Date"], dayfirst=True)
    screen.rename(columns={"Time": "Minutes"}, inplace=True)
    daily_minutes = (
        screen.groupby("Date")["Minutes"]
        .sum()
        .reset_index(name="total_minutes")
    )

    spend = pd.read_csv(spend_path, sep=";")
    if "Date" not in spend.columns:
        raise ValueError("Missing 'Date' column in spending CSV")
    spend["Date"] = pd.to_datetime(spend["Date"], dayfirst=True)
    amount_col = next(
        (c for c in spend.columns if c.lower() in
         ["amount", "total_spending", "price", "total", "transaction amount", "miktar"]),
        None,
    )
    if amount_col is None:
        raise ValueError("Could not infer amount column in spending CSV")
    daily_spend = (
        spend.groupby("Date")[amount_col]
        .sum()
        .reset_index(name="daily_spend")
    )

    df = pd.merge(daily_minutes, daily_spend, on="Date", how="inner")
    if df.empty:
        raise ValueError("No overlapping dates between datasets")

    df["day_of_week"] = df["Date"].dt.day_name()
    df["is_weekend"] = df["Date"].dt.dayofweek >= 5

    df["prev_minutes"] = df["total_minutes"].shift(1)
    df["minutes_trend"] = df["total_minutes"].rolling(3).mean().shift(1)
    df["prev_spend"] = df["daily_spend"].shift(1)
    df["spend_trend"] = df["daily_spend"].rolling(3).mean().shift(1)

    df = df.fillna(
        {
            "prev_minutes": df["total_minutes"].mean(),
            "minutes_trend": df["total_minutes"].mean(),
            "prev_spend": df["daily_spend"].mean(),
            "spend_trend": df["daily_spend"].mean(),
        }
    )
    return df

def raw_visuals(df: pd.DataFrame):
    out0 = ROOT_OUT / "0_raw_inspection"

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, x="total_minutes", y="daily_spend",
        hue="is_weekend", palette="viridis", alpha=0.8)
    sns.regplot(
        data=df, x="total_minutes", y="daily_spend",
        scatter=False, color="red", line_kws={"lw": 2, "ls": "--"})
    plt.title("Screen-Time vs Daily Spend")
    plt.tight_layout()
    plt.savefig(out0 / "scatter_minutes_spend.png"); plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="day_of_week", y="daily_spend", data=df,
                order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    plt.title("Daily Spend by Weekday")
    plt.tight_layout()
    plt.savefig(out0 / "box_dayofweek_spend.png"); plt.close()

    num_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
    plt.title("Correlation Matrix (numeric)")
    plt.tight_layout()
    plt.savefig(out0 / "corr_matrix.png"); plt.close()

def time_series_visuals(df: pd.DataFrame):
    out1 = ROOT_OUT / "1_time_series"

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(df["Date"], df["total_minutes"], label="Screen Minutes")
    ax1.set_ylabel("Minutes")
    ax2 = ax1.twinx()
    ax2.plot(df["Date"], df["daily_spend"], label="Daily Spend", color="coral")
    ax2.set_ylabel("Spend")
    ax1.set_title("Time-Series: Screen Minutes & Spend")
    ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.tight_layout(); plt.savefig(out1 / "timeseries_minutes_spend.png"); plt.close()

    df["roll_corr"] = df["total_minutes"].rolling(30).corr(df["daily_spend"])
    plt.figure(figsize=(12, 5))
    plt.plot(df["Date"], df["roll_corr"])
    plt.title("30-Day Rolling Correlation (Minutes vs Spend)")
    plt.ylabel("Pearson r"); plt.ylim(-1, 1)
    plt.axhline(0, color="gray", ls="--")
    plt.tight_layout(); plt.savefig(out1 / "rolling_corr_30d.png"); plt.close()

def prepare_ml(df):
    y = df["daily_spend"]
    cat_features = ["day_of_week", "is_weekend"]
    num_features = [c for c in df.columns if c not in ["Date", "daily_spend"] + cat_features]

    # Handle potential infinite values and outliers
    for col in num_features:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        df[col] = df[col].fillna(df[col].mean())
        
        # Clip outliers to 3 standard deviations
        mean = df[col].mean()
        std = df[col].std()
        df[col] = df[col].clip(lower=mean - 3*std, upper=mean + 3*std)

    X = df[cat_features + num_features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)

    return X_train, X_test, y_train, y_test, cat_features, num_features

def build_models(X_tr, X_te, y_tr, y_te, cat_feats, num_feats):
    out2 = ROOT_OUT / "2_modeling"
    
    # Simplified preprocessing
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preproc = ColumnTransformer([
        ("num", num_pipe, num_feats),
        ("cat", cat_pipe, cat_feats)
    ])

    # Only use tree-based models which are more robust to numerical issues
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=250, random_state=42),
        "GradientBoost": GradientBoostingRegressor(n_estimators=250, random_state=42),
    }

    results, feat_imps = {}, {}
    for name, mdl in models.items():
        pipe = Pipeline([("pre", preproc), ("model", mdl)])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        mse = mean_squared_error(y_te, preds)
        results[name] = {
            "RMSE": np.sqrt(mse),
            "MAE": mean_absolute_error(y_te, preds),
            "R2": r2_score(y_te, preds),
            "pipe": pipe,
            "preds": preds,
        }
        if hasattr(mdl, "feature_importances_"):
            preproc.fit(X_tr)
            cat_names = preproc.named_transformers_["cat"].named_steps["oh"].get_feature_names_out(cat_feats)
            fnames = np.concatenate([num_feats, cat_names])
            imps = mdl.feature_importances_
            top = np.argsort(imps)[::-1][:10]
            feat_imps[name] = {fnames[i]: imps[i] for i in top}

    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(results.keys()), y=[r["R2"] for r in results.values()], 
                hue=list(results.keys()), palette="Blues_d", legend=False)
    plt.ylabel("R²"); plt.xticks(rotation=45, ha="right")
    plt.title("Model R² Scores")
    plt.tight_layout(); plt.savefig(out2 / "model_r2_scores.png"); plt.close()

    best_name = max(results, key=lambda k: results[k]["R2"])
    best_res  = results[best_name]
    best_pipe = best_res["pipe"]
    print(f"\nBest model → {best_name}  (R² = {best_res['R2']:.3f})")

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_te, y=best_res["preds"], alpha=0.7)
    plt.plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], ls="--", color="red")
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.title("Actual vs Predicted")
    plt.tight_layout(); plt.savefig(out2 / "actual_vs_pred.png"); plt.close()

    residuals = y_te - best_res["preds"]
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=30)
    plt.title("Residual Distribution (Best Model)")
    plt.xlabel("Residual (Actual – Pred)"); plt.tight_layout()
    plt.savefig(out2 / "residuals_hist.png"); plt.close()

    err_df = pd.DataFrame({"weekday": X_te["day_of_week"], "abs_err": residuals.abs()})
    plt.figure(figsize=(8, 5))
    sns.barplot(x="weekday", y="abs_err", data=err_df,
                order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    plt.title("Absolute Prediction Error by Weekday")
    plt.ylabel("Abs Error"); plt.tight_layout()
    plt.savefig(out2 / "error_by_weekday.png"); plt.close()

    if best_name in feat_imps:
        plt.figure(figsize=(8, 6))
        sns.barplot(
            x=list(feat_imps[best_name].values()),
            y=list(feat_imps[best_name].keys()),
            hue=list(feat_imps[best_name].keys()),
            palette="magma",
            legend=False)
        plt.title(f"{best_name} – Top Feature Importances")
        plt.tight_layout(); plt.savefig(out2 / "best_model_feature_importance.png"); plt.close()

    if _SHAP_AVAILABLE:
        EXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
        tree_model = best_pipe.named_steps["model"]

        preproc = best_pipe.named_steps["pre"]
        X_te_t = preproc.transform(X_te)
        cat_names = preproc.named_transformers_["cat"].named_steps["oh"].get_feature_names_out(cat_feats)
        feature_names = np.concatenate([num_feats, cat_names])

        explainer = shap.TreeExplainer(tree_model)
        shap_vals = explainer.shap_values(X_te_t)
        shap.summary_plot(shap_vals, features=X_te_t, feature_names=feature_names, show=False)
        plt.tight_layout(); plt.savefig(EXPLAIN_DIR / "shap_summary.png"); plt.close()

    model_path = ROOT_OUT / "best_screen_spend_model.pkl"
    joblib.dump(best_pipe, model_path)

    meta = {
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "best_model": best_name,
        "r2": best_res["R2"],
        "rmse": best_res["RMSE"],
        "feature_columns": {
            "categorical": cat_feats,
            "numerical": num_feats,
        },
        "input_files": {
            "screen_csv": str(args.screen),
            "spend_csv": str(args.spend),
        },
        "model_path": str(model_path),
        "shap_generated": _SHAP_AVAILABLE,
        "version": "1.0.0",
    }
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, required=True, help="Screentime CSV path")
    parser.add_argument("--spend", type=Path, required=True, help="Spending CSV path")
    args = parser.parse_args()

    df = load_and_enrich_data(args.screen, args.spend)
    print(f"Dataset merged: {df.shape[0]} rows")

    raw_visuals(df)
    time_series_visuals(df)

    X_tr, X_te, y_tr, y_te, cat_f, num_f = prepare_ml(df)
    feature_selection_rfe(X_tr, y_tr, cat_f, num_f)

    build_models(X_tr, X_te, y_tr, y_te, cat_f, num_f)

    print(f"All outputs saved under '{ROOT_OUT}/' directory.")

def feature_selection_rfe(X_tr, y_tr, cat_feats, num_feats):
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", num_pipe, num_feats), ("cat", cat_pipe, cat_feats)])
    X_tr_t = pre.fit_transform(X_tr)
    cat_names = pre.named_transformers_["cat"].named_steps["oh"].get_feature_names_out(cat_feats) if cat_feats else []
    feat_names = np.concatenate([num_feats, cat_names])

    rf = RandomForestRegressor(n_estimators=250, random_state=42)
    for n in [5, 8, 10]:
        if n > len(feat_names):
            continue
        selector = RFE(rf, n_features_to_select=n).fit(X_tr_t, y_tr)
        sel = feat_names[selector.support_]
        print(f"RFE top {n} features: {', '.join(sel)}")

if __name__ == "__main__":
    main()