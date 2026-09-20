from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "churn_data.csv"
RESULT_PATH = ROOT / "results" / "model_metrics.csv"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["churn"])
    y = df["churn"]

    numeric_features = [
        "tenure_months",
        "monthly_charges",
        "support_tickets",
    ]
    categorical_features = [
        "contract_type",
        "payment_method",
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
    }

    rows = []

    for model_name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        rows.append({
            "model": model_name,
            "accuracy": round(
                accuracy_score(y_test, predictions), 3
            ),
            "precision": round(
                precision_score(y_test, predictions, zero_division=0), 3
            ),
            "recall": round(
                recall_score(y_test, predictions, zero_division=0), 3
            ),
            "f1_score": round(
                f1_score(y_test, predictions, zero_division=0), 3
            ),
        })

    results = pd.DataFrame(rows)
    RESULT_PATH.parent.mkdir(exist_ok=True)
    results.to_csv(RESULT_PATH, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved results to: {RESULT_PATH}")

if __name__ == "__main__":
    main()
