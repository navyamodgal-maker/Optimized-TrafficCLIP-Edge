"""
Assemble outputs/results.txt, outputs/evaluation_results.txt, and a
results/figures + results/metrics folder structure from the numbers
already produced by train_proposed.py, evaluate_proposed.py, and
qat_train.py -- mirroring the senior's OptimizedTrafficCLIP repo's
output layout, at the scale of our actual (single-config) experiment.

QAT numbers are hardcoded from the printed console output since
qat_train.py doesn't currently save them to a file.
"""

import os
import json
import shutil

OUTPUT_DIR = "output"
RESULTS_DIR = "results"

# From the printed QAT console output -- qat_train.py doesn't save
# these to a file, so they're transcribed here rather than re-derived.
QAT_RESULTS = {
    "fp32_baseline_accuracy": 0.9963,
    "naive_fake_quant_accuracy": 0.9963,
    "qat_finetuned_accuracy": 0.9985,
}


def main():
    with open(os.path.join(OUTPUT_DIR, "results_summary.json")) as f:
        summary = json.load(f)

    standard = summary["standard_evaluation"]
    open_set = summary.get("open_set_evaluation")

    class_names = standard["class_names"]
    precision = standard["per_class_precision"]
    recall = standard["per_class_recall"]
    f1 = standard["per_class_f1"]
    support = standard["per_class_support"]
    cm = standard["confusion_matrix"]
    accuracy = standard["accuracy"]
    macro_precision = standard["precision"]
    macro_recall = standard["recall"]
    macro_f1 = standard["f1"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_txt_lines = [
        "ProposedTrafficCLIP -- Training & Optimization Summary",
        "=" * 70,
        "",
        "--- train_proposed.py (closed-set, all 10 classes) ---",
        "Dataset: USTC-TFC2016, 900 flow samples/class, 9,000 total",
        f"Classes: {class_names}",
        "Epochs: 10 | Batch size: 16",
        "Split: 70% train / 15% val / 15% test (seed=42)",
        "",
        "--- train_proposed.py --open-set (Zeus, BitTorrent held out) ---",
        "Training samples: 5,040 | Val: 1,080 | Test: 1,080",
        f"Held-out (unknown) classes: {open_set['held_out_classes'] if open_set else '[]'}",
        "",
        "--- qat_train.py (fusion head only, 5 epochs) ---",
        f"FP32 baseline accuracy              : {QAT_RESULTS['fp32_baseline_accuracy']:.4f}",
        f"Naive fake-quant, no fine-tuning     : {QAT_RESULTS['naive_fake_quant_accuracy']:.4f}",
        f"Fake-quant AFTER QAT fine-tuning     : {QAT_RESULTS['qat_finetuned_accuracy']:.4f}",
        "",
        "Note: naive fake-quantization showed no accuracy drop from FP32",
        "in this run, so the post-fine-tuning improvement reflects a few",
        "additional epochs of fusion-head training rather than recovery",
        "from a quantization-induced accuracy collapse.",
        "",
    ]
    with open(os.path.join(OUTPUT_DIR, "results.txt"), "w") as f:
        f.write("\n".join(results_txt_lines))
    print(f"Wrote {OUTPUT_DIR}/results.txt")

    lines = []
    lines.append("ProposedTrafficCLIP TEST RESULTS")
    lines.append("=" * 70)
    lines.append(f"Test samples: {sum(support)}")
    lines.append(f"Accuracy: {accuracy:.4f}")
    lines.append(f"Macro Precision: {macro_precision:.4f}")
    lines.append(f"Macro Recall: {macro_recall:.4f}")
    lines.append(f"Macro F1: {macro_f1:.4f}")
    lines.append("")
    lines.append("Classification Report")
    lines.append("=" * 70)
    lines.append(f"{'':>14}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>12}")
    lines.append("")
    for name, p, r, f_, s in zip(class_names, precision, recall, f1, support):
        lines.append(f"{name:>14}{p:>12.2f}{r:>12.2f}{f_:>12.2f}{s:>12}")
    lines.append("")
    lines.append(f"{'accuracy':>14}{'':>12}{'':>12}{accuracy:>12.2f}{sum(support):>12}")
    lines.append(f"{'macro avg':>14}{macro_precision:>12.2f}{macro_recall:>12.2f}{macro_f1:>12.2f}{sum(support):>12}")
    lines.append("")
    lines.append("Confusion Matrix")
    lines.append("=" * 70)
    lines.append("Rows = true class, Columns = predicted class, order matches list above")
    lines.append(f"Classes: {class_names}")
    for row in cm:
        lines.append("[" + " ".join(f"{v:4d}" for v in row) + "]")
    lines.append("")

    if open_set:
        lines.append("")
        lines.append("Open-Set / Zero-Day Detection")
        lines.append("=" * 70)
        lines.append(f"Held-out (unknown) classes: {open_set['held_out_classes']}")
        lines.append(f"ROC-AUC (known vs unknown): {open_set['auc']:.4f}")
        lines.append("")

    lines.append("Quantization-Aware Training (fusion head)")
    lines.append("=" * 70)
    lines.append(f"FP32 baseline accuracy              : {QAT_RESULTS['fp32_baseline_accuracy']:.4f}")
    lines.append(f"Naive fake-quant, no fine-tuning     : {QAT_RESULTS['naive_fake_quant_accuracy']:.4f}")
    lines.append(f"Fake-quant AFTER QAT fine-tuning     : {QAT_RESULTS['qat_finetuned_accuracy']:.4f}")

    with open(os.path.join(OUTPUT_DIR, "evaluation_results.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUTPUT_DIR}/evaluation_results.txt")

    figures_dir = os.path.join(RESULTS_DIR, "figures")
    metrics_dir = os.path.join(RESULTS_DIR, "metrics")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    for fname in ["confusion_matrix.png", "per_class_metrics.png", "open_set_roc_curve.png"]:
        src = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(figures_dir, fname))

    with open(os.path.join(metrics_dir, "per_class_results.csv"), "w") as f:
        f.write("class,precision,recall,f1,support\n")
        for name, p, r, f_, s in zip(class_names, precision, recall, f1, support):
            f.write(f"{name},{p:.4f},{r:.4f},{f_:.4f},{s}\n")

    with open(os.path.join(metrics_dir, "summary_results.csv"), "w") as f:
        f.write("metric,value\n")
        f.write(f"closed_set_accuracy,{accuracy:.4f}\n")
        f.write(f"closed_set_macro_f1,{macro_f1:.4f}\n")
        f.write(f"closed_set_macro_precision,{macro_precision:.4f}\n")
        f.write(f"closed_set_macro_recall,{macro_recall:.4f}\n")
        if open_set:
            f.write(f"open_set_roc_auc,{open_set['auc']:.4f}\n")
        f.write(f"qat_fp32_baseline,{QAT_RESULTS['fp32_baseline_accuracy']:.4f}\n")
        f.write(f"qat_naive_fake_quant,{QAT_RESULTS['naive_fake_quant_accuracy']:.4f}\n")
        f.write(f"qat_finetuned,{QAT_RESULTS['qat_finetuned_accuracy']:.4f}\n")

    print(f"Wrote {metrics_dir}/per_class_results.csv")
    print(f"Wrote {metrics_dir}/summary_results.csv")
    print(f"Copied figures to {figures_dir}/")


if __name__ == "__main__":
    main()
