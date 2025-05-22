"""
How to run
----------
python3 ml_methods.py --screen data/screentime.csv --spend data/spending.csv

"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression, Ridge)
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# DATA PREP
def load_and_prepare(screen_path: Path, spend_path: Path) -> pd.DataFrame:
    """Load raw CSVs and return a merged daily-level DataFrame."""
    try:
        screen = pd.read_csv(screen_path, sep=';')
        if "Date" not in screen.columns:
            raise ValueError("Expected a 'Date' column in screentime file")
        screen["Date"] = pd.to_datetime(screen["Date"], dayfirst=True)
        screen.rename(columns={"Time": "Minutes"}, inplace=True)

        if "Category" in screen.columns:
            screen_cat = (
                screen.pivot_table(
                    index="Date",
                    columns="Category",
                    values="Minutes",
                    aggfunc="sum",
                )
                .add_prefix("cat_")
                .fillna(0)
            )
            screen_daily = screen.groupby("Date")["Minutes"].sum().to_frame(
                "total_minutes"
            )
            screen = screen_daily.join(screen_cat, how="left").reset_index()
        else:
            screen = (
                screen.groupby("Date")["Minutes"]
                .sum()
                .reset_index(name="total_minutes")
            )

        spending = pd.read_csv(spend_path, sep=';')
        if "Date" not in spending.columns:
            raise ValueError("Expected a 'Date' column in spending file")
        spending["Date"] = pd.to_datetime(spending["Date"], dayfirst=True)

        possible_amount_cols = [
            "Amount",
            "Transaction Amount",
            "Price",
            "Total",
            "Miktar",
            "Total_Spending",
        ]
        amount_col = next(
            (c for c in spending.columns if c.strip() in possible_amount_cols), None
        )
        if amount_col is None:
            raise ValueError(
                "Could not find a valid amount column "
                "(e.g., Amount, Price, Total, Total_Spending)"
            )

        spending_daily = (
            spending.groupby("Date")[amount_col]
            .sum()
            .reset_index(name="daily_spend")
        )

        df = pd.merge(screen, spending_daily, on="Date", how="inner")
        if len(df) == 0:
            raise ValueError("No matching dates found between screen time and spending data")
            
        df["day_of_week"] = df["Date"].dt.dayofweek  # 0=Mon
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        if df["daily_spend"].isnull().any() or df["total_minutes"].isnull().any():
            print("Warning: Found missing values in the data. These will be dropped.")
            
        return df.dropna(subset=["daily_spend", "total_minutes"])
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: One or both input files are empty", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing data: {e}", file=sys.stderr)
        sys.exit(1)


# MODEL TRAIN / EVAL
def build_models(df: pd.DataFrame) -> dict:
    """Train several regressors and return metric dict."""
    target = "daily_spend"
    feature_cols = [c for c in df.columns if c not in ["Date", target]]

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    preproc = ColumnTransformer(
        [("num", StandardScaler(), X.columns.tolist())]
    )

    model_dict = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.01),
        "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
        "RandomForest": RandomForestRegressor(
            n_estimators=250, random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, random_state=42
        ),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(kernel="rbf", C=100, epsilon=0.1),
    }

    results = {}
    for name, base_model in model_dict.items():
        try:
            model = Pipeline([("pre", preproc), ("model", base_model)])
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            results[name] = {
                "MAE": mean_absolute_error(y_test, preds),
                "R2": r2_score(y_test, preds),
            }
        except Exception as e:
            print(f"Warning: Failed to train {name} model: {e}", file=sys.stderr)
            continue
            
    if not results:
        raise RuntimeError("No models were successfully trained")
        
    return results


# CLI 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply multiple ML models to screen-time vs spending data"
    )
    parser.add_argument(
        "--screen", type=Path, default="screentime.csv", help="Screen-time CSV"
    )
    parser.add_argument(
        "--spend", type=Path, default="spending.csv", help="Spending CSV"
    )
    args = parser.parse_args()

    try:
        df = load_and_prepare(args.screen, args.spend)
        print(f"Merged dataset: {df.shape[0]} days, {df.shape[1]-2} features")

        results = build_models(df)
        print("\nModel performance (test set):")
        for name, m in sorted(results.items(), key=lambda kv: kv[1]["MAE"]):
            print(f"{name:16}: MAE = {m['MAE']:.2f}, R² = {m['R2']:.3f}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()