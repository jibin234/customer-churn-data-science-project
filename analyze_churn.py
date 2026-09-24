"""Explore synthetic customer churn data and compare two baseline classifiers."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "customer_churn.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print("\nMissing values:\n", df.isna().sum())
    print("\nChurn rate:", f"{(df.churn == 'Yes').mean():.1%}")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="contract_type", hue="churn", ax=ax)
    ax.set(title="Synthetic customer churn by contract", xlabel="Contract type", ylabel="Customers")
    fig.tight_layout()
    fig.savefig(OUT / "churn_by_contract.png", dpi=160)
    plt.close(fig)

    X = df.drop(columns=["customer_id", "churn"])
    y = (df["churn"] == "Yes").astype(int)
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    prep = ColumnTransformer([
        ("numeric", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("fill", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    }
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    lines = ["Evaluation on a held-out 25% test set", "Note: metrics describe this synthetic teaching dataset only.", ""]
    for name, estimator in models.items():
        model = Pipeline([("preprocess", prep), ("model", estimator)])
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        lines.append(name)
        lines.append(f"Accuracy:  {accuracy_score(y_test, pred):.3f}")
        lines.append(f"Precision: {precision_score(y_test, pred, zero_division=0):.3f}")
        lines.append(f"Recall:    {recall_score(y_test, pred, zero_division=0):.3f}")
        lines.append(f"F1 score:  {f1_score(y_test, pred, zero_division=0):.3f}")
        lines.append("Confusion matrix [ [TN, FP], [FN, TP] ]:")
        lines.append(str(confusion_matrix(y_test, pred).tolist()))
        lines.append("")
    report = "\n".join(lines)
    (OUT / "model_metrics.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nSaved chart and metrics in {OUT}")

if __name__ == "__main__":
    main()
