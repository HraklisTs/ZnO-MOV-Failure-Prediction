"""Generate synthetic waveform data for a public ML demonstration."""

import csv
import math
import random
from pathlib import Path

SEED = 42
SAMPLES_PER_WAVEFORM = 16
WAVEFORM_COUNT = 200


def generate_dataset(output_path: Path) -> None:
    rng = random.Random(SEED)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_id", "label"]
            + [f"voltage_{i:02d}" for i in range(SAMPLES_PER_WAVEFORM)]
        )

        for sample_id in range(WAVEFORM_COUNT):
            label = "FAIL" if sample_id % 2 == 0 else "PASS"
            amplitude = rng.uniform(0.85, 1.15)

            if label == "FAIL":
                amplitude *= 1.35

            waveform = [
                round(
                    amplitude
                    * math.exp(-2.5 * i / SAMPLES_PER_WAVEFORM)
                    * math.sin(math.pi * i / (SAMPLES_PER_WAVEFORM - 1))
                    + rng.gauss(0, 0.025),
                    6,
                )
                for i in range(SAMPLES_PER_WAVEFORM)
            ]

            writer.writerow([sample_id, label] + waveform)


if __name__ == "__main__":
    destination = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "synthetic"
        / "waveforms.csv"
    )
    generate_dataset(destination)
    print(f"Synthetic dataset created: {destination}")
