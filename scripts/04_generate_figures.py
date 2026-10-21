"""
Regenerate the 20-class figures from saved result files -- no model,
dataset or GPU needed.

Reads:  output/results_summary.json, output/training_logs.json,
        output/qat_results.json
Writes: output/confusion_matrix.png, output/confusion_matrix_normalized.png,
        output/per_class_metrics.png, output/training_curves.png,
        output/qat_comparison.png

The open-set ROC curve (output/open_set_roc_curve.png) is NOT redrawn here:
the per-sample scores it needs are only produced inside
`python -m src.evaluate_proposed`, which also writes the PNG.

Usage (from the project root):
    python scripts/04_generate_figures.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.utils.plotting import (
    plot_confusion_matrix,
    plot_per_class_bars,
    plot_training_curves,
    plot_qat_comparison,
)

OUTPUT_DIR = "output"


def main():
    with open(os.path.join(OUTPUT_DIR, "results_summary.json")) as f:
        standard = json.load(f)["standard_evaluation"]

    names = standard["class_names"]
    cm = np.array(standard["confusion_matrix"])
    assert cm.shape == (len(names), len(names)), "confusion matrix / class count mismatch"

    plot_confusion_matrix(cm, names, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plot_confusion_matrix(cm, names, os.path.join(OUTPUT_DIR, "confusion_matrix_normalized.png"),
                          normalize=True)
    plot_per_class_bars(
        names,
        standard["per_class_precision"],
        standard["per_class_recall"],
        standard["per_class_f1"],
        os.path.join(OUTPUT_DIR, "per_class_metrics.png"),
    )

    logs_path = os.path.join(OUTPUT_DIR, "training_logs.json")
    if os.path.exists(logs_path):
        with open(logs_path) as f:
            logs = json.load(f)
        plot_training_curves(logs.get("closed_set"), logs.get("open_set"),
                             os.path.join(OUTPUT_DIR, "training_curves.png"))

    qat_path = os.path.join(OUTPUT_DIR, "qat_results.json")
    if os.path.exists(qat_path):
        with open(qat_path) as f:
            plot_qat_comparison(json.load(f), os.path.join(OUTPUT_DIR, "qat_comparison.png"))


if __name__ == "__main__":
    main()
