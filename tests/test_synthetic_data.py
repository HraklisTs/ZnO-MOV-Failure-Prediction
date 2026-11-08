import csv
from src.generate_synthetic_data import generate_dataset

def test_synthetic_dataset(tmp_path):
    output = tmp_path / "waveforms.csv"
    generate_dataset(output)
    with output.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 200
    assert len(rows[0]) == 18
    assert {row["label"] for row in rows} == {"PASS", "FAIL"}
