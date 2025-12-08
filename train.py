import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn import set_config

RANDOM_SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATA_DIR / "Clean_Dataset.csv"
MODEL_PATH = MODELS_DIR / "model.joblib"

def load_and_preprocess_raw(path: Path) -> pd.DataFrame:
    """
    Load the cleaned flight dataset and apply basic preprocessing:
    - Drop technical index column (if present)
    - Map 'stops' from category labels to numeric codes
    - Construct a 'route' feature from source and destination
    - Create 'log_price' as the regression target
    """
    df = pd.read_csv(path)

    # Drop technical index column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df_processed = df.copy()

    # Map stops (zero/one/two_or_more → 0/1/2)
    stops_mapping = {"zero": 0, "one": 1, "two_or_more": 2}
    df_processed["stops"] = df_processed["stops"].map(stops_mapping)

    # Create route feature
    df_processed["route"] = (
        df_processed["source_city"] + "-" + df_processed["destination_city"]
    )

    # Create log_price target to stabilize variance and handle skewness
    df_processed["log_price"] = np.log1p(df_processed["price"])

    return df_processed


def build_X_y(df_processed: pd.DataFrame):
    """
    Build feature matrix X and target y.

    - Exclude original price, target column, and purely technical fields
    - Use 'log_price' as the regression target
    """
    drop_cols = ["price", "log_price", "flight", "source_city", "destination_city"]
    drop_cols = [c for c in drop_cols if c in df_processed.columns]
    X = df_processed.drop(columns=drop_cols)
    y = df_processed["log_price"].values  # log(price)

    # Ensure days_left is treated as numeric
    if "days_left" in X.columns:
        X["days_left"] = X["days_left"].astype(float)

    return X, y


def split_data(X, y):
    """
    Split the dataset into train and test sets.

    Returns both the log-price targets and the original-scale targets
    (for metrics computed on the original price scale).
    """
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )
    # For evaluation (MAE/R2) we convert back to original price
    y_test_orig = np.expm1(y_test_log)
    return X_train, X_test, y_train_log, y_test_log, y_test_orig


def build_preprocessor(X: pd.DataFrame):
    """
    Build the preprocessing pipeline:

    - Numerical features: ['duration', 'days_left']
    - Categorical features: all remaining columns
    - OneHotEncoder with dense output
    - ColumnTransformer configured to output a pandas DataFrame
    """
    numerical_features = ["duration", "days_left"]
    categorical_features = [col for col in X.columns if col not in numerical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            )
        ],
        remainder="passthrough",
    )

    # Enable metadata routing and ensure the transformer returns a DataFrame
    set_config(enable_metadata_routing=True)
    preprocessor.set_output(transform="pandas")
    preprocessor.verbose_feature_names_out = False

    print("--- Feature Definition ---")
    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)

    return preprocessor


def get_models():
    """
    Define a collection of candidate regression models with
    reasonable baseline hyperparameters.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_SEED),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            tree_method="hist",
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            random_state=RANDOM_SEED,
        ),
    }
    return models


def train_and_compare_models(
    preprocessor, X_train, X_test, y_train_log, y_test_log, y_test_orig
):
    """
    Train and evaluate all candidate models, then:

    - Print performance (R² and MAE) on the original price scale
    - Return the best-performing pipeline (based on MAE)
    """
    models = get_models()
    results = []

    best_model_name = None
    best_mae = float("inf")
    best_pipeline = None

    print("Starting model comparison loop...\n")

    for name, model in models.items():
        print(f"Training model: {name}")

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train_log)

        # Predict in log-price space
        y_pred_log = pipeline.predict(X_test)
        # Convert predictions back to the original price scale
        y_pred_orig = np.expm1(y_pred_log)

        # Evaluate performance on the original price scale
        r2 = r2_score(y_test_orig, y_pred_orig)
        mae = mean_absolute_error(y_test_orig, y_pred_orig)

        print(f"  -> R2: {r2:.4f}, MAE: {mae:.2f}")

        results.append(
            {
                "Model": name,
                "R2 Score": r2,
                "MAE": mae,
                "Pipeline": pipeline,
            }
        )

        if mae < best_mae:
            best_mae = mae
            best_model_name = name
            best_pipeline = pipeline

    print("\n=== Model Comparison Complete ===")
    for r in sorted(results, key=lambda x: x["MAE"]):
        print(f"{r['Model']:>16} | R2={r['R2 Score']:.4f} | MAE={r['MAE']:.2f}")

    print(f"\nBest model: {best_model_name} (MAE={best_mae:.2f})")

    return best_pipeline, results


def main():
    np.random.seed(RANDOM_SEED)

    print(f"Loading data from {DATA_PATH}")
    df_processed = load_and_preprocess_raw(DATA_PATH)
    X, y = build_X_y(df_processed)
    X_train, X_test, y_train_log, y_test_log, y_test_orig = split_data(X, y)

    preprocessor = build_preprocessor(X_train)
    best_pipeline, results = train_and_compare_models(
        preprocessor, X_train, X_test, y_train_log, y_test_log, y_test_orig
    )

    if best_pipeline is not None:
        joblib.dump(best_pipeline, MODEL_PATH)
        print(f"Saved best model pipeline to {MODEL_PATH}")



if __name__ == "__main__":
    main()
