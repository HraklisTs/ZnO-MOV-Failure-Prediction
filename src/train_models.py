"""Train and evaluate ML models using synthetic MOV waveforms."""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "synthetic" / "waveforms.csv"
RANDOM_STATE = 42


def extract_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Extract illustrative statistical features from 16-point waveforms."""
    voltage_columns = [f"voltage_{i:02d}" for i in range(16)]
    waveforms = frame[voltage_columns].to_numpy(dtype=float)

    features = pd.DataFrame({
        "peak": np.max(waveforms, axis=1),
        "minimum": np.min(waveforms, axis=1),
        "mean": np.mean(waveforms, axis=1),
        "std": np.std(waveforms, axis=1),
        "rms": np.sqrt(np.mean(waveforms ** 2, axis=1)),
        "energy": np.sum(waveforms ** 2, axis=1),
        "peak_to_peak": np.ptp(waveforms, axis=1),
        "absolute_mean": np.mean(np.abs(waveforms), axis=1),
    })

    return features


def main() -> None:
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

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        print(f"\n{name}")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
        print(
            classification_report(
                y_test,
                predictions,
                labels=["FAIL", "PASS"],
                zero_division=0,
            )
        )


if __name__ == "__main__":
    main()
