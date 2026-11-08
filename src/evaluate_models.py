"""Evaluate synthetic ZnO MOV models and generate publication-ready visualizations."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from train_models import DATA_PATH, RANDOM_STATE, extract_features

ROOT = Path(__file__).resolve().parents[1]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Synthetic dataset missing. Run generate_synthetic_data.py first."
        )

    frame = pd.read_csv(DATA_PATH)
    X = extract_features(frame)
    y = frame["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", random_state=RANDOM_STATE),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }

    output_dir = ROOT / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    random_forest = None

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        report = classification_report(
            y_test,
            predictions,
            labels=["FAIL", "PASS"],
            output_dict=True,
            zero_division=0,
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": report["macro avg"]["precision"],
            "Recall": report["macro avg"]["recall"],
            "F1-score": report["macro avg"]["f1-score"],
        })

        fig, ax = plt.subplots(figsize=(7, 6))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            labels=["FAIL", "PASS"],
            display_labels=["FAIL", "PASS"],
            cmap="Blues",
            values_format="d",
            ax=ax,
            colorbar=False,
        )

        ax.set_title(f"{name} - Synthetic Test Data")
        fig.tight_layout()
        fig.savefig(
            output_dir / f"confusion_matrix_{name.lower().replace(' ', '_')}.png",
            dpi=250,
            bbox_inches="tight",
        )
        plt.close(fig)

        if name == "Random Forest":
            random_forest = model

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "model_metrics.csv", index=False)

    # Model performance comparison
    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
    model_names = results_df["Model"].tolist()

    x = np.arange(len(model_names))
    width = 0.19

    fig, ax = plt.subplots(figsize=(11, 7))

    for i, metric in enumerate(metrics):
        values = results_df[metric].to_numpy()
        positions = x + (i - 1.5) * width

        bars = ax.bar(
            positions,
            values,
            width,
            label=metric,
        )

        ax.bar_label(
            bars,
            labels=[f"{v:.2%}" for v in values],
            padding=3,
            fontsize=8,
            rotation=90,
        )

    ax.set_title(
        "Model Performance Comparison - Synthetic Data",
        fontsize=15,
        pad=18,
    )
    ax.set_ylabel("Score")
    ax.set_xlabel("Classification Model")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.17)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(1.0)
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        frameon=False,
    )
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.text(
        0.5,
        0.015,
        "All metrics are calculated on the same synthetic test set.",
        ha="center",
        fontsize=9,
        color="dimgray",
    )

    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.savefig(
        output_dir / "model_comparison.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Random Forest feature importance
    importance = pd.Series(
        random_forest.feature_importances_,
        index=X.columns,
    ).sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.barh(
        importance.index,
        importance.values,
    )

    ax.bar_label(
        bars,
        labels=[f"{v:.4f}" for v in importance.values],
        padding=4,
        fontsize=9,
    )

    ax.set_title(
        "Random Forest Feature Importance - Synthetic Data",
        fontsize=14,
        pad=15,
    )
    ax.set_xlabel("Feature Importance")
    ax.set_xlim(0, max(importance.values) * 1.18)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(
        output_dir / "feature_importance.png",
        dpi=250,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nSynthetic model evaluation completed.")
    print(results_df.to_string(index=False))
    print("\nGenerated files:")
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            print(f" - {path.name}")

    print(f"\nOutput directory: {output_dir}")


if __name__ == "__main__":
    main()