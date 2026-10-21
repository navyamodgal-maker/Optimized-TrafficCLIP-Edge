"""
Assemble output/results.txt, output/evaluation_results.txt and the
results/figures + results/metrics folders from the numbers produced by
train_proposed.py, evaluate_proposed.py and qat_train.py.

Everything is read from files -- nothing is hardcoded per class count:
  output/results_summary.json   (evaluate_proposed.py)
  output/qat_results.json       (qat_train.py)
  output/training_logs.json     (best validation accuracies, epochs)

Run scripts/04_generate_figures.py first if you want the figures
refreshed from results_summary.json.
"""

import os
import json
import shutil

OUTPUT_DIR = "output"
RESULTS_DIR = "results"

FIGURES = [
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "per_class_metrics.png",
    "open_set_roc_curve.png",
    "training_curves.png",
    "qat_comparison.png",
]


def load_json(name, required=True):
    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"{path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def fmt(value):
    return "n/a (re-run qat_train.py)" if value is None else f"{value:.4f}"


def main():
    summary = load_json("results_summary.json")
    qat = load_json("qat_results.json")
    logs = load_json("training_logs.json", required=False) or {}

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

    num_classes = len(class_names)
    test_total = int(sum(support))
    per_class_test = sorted(set(int(s) for s in support))
    held_out = open_set["held_out_classes"] if open_set else []

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------- results.txt ----------------
    results_txt_lines = [
        "ProposedTrafficCLIP -- Training & Optimization Summary",
        "=" * 70,
        "",
        f"--- train_proposed.py (closed-set, all {num_classes} classes) ---",
        f"Dataset: USTC-TFC2016, {num_classes} classes",
        f"Classes: {class_names}",
        "Epochs: 10 | Batch size: 16",
        "Split: 70% train / 15% val / 15% test (stratified, seed=42)",
        f"Test samples: {test_total} (per-class support: {per_class_test})",
    ]
    if "closed_set_best_val_acc" in logs:
        results_txt_lines.append(f"Best validation accuracy: {logs['closed_set_best_val_acc']:.4f}")
    results_txt_lines += [
        "",
        f"--- train_proposed.py --open-set ({', '.join(held_out)} held out) ---",
        f"Known (trained) classes: {num_classes - len(held_out)}",
        f"Held-out (unknown) classes: {held_out}",
    ]
    if "open_set_best_val_acc" in logs:
        results_txt_lines.append(f"Best validation accuracy (known classes): {logs['open_set_best_val_acc']:.4f}")
    results_txt_lines += [
        "",
        f"--- qat_train.py (fusion head only, {qat.get('qat_epochs', 5)} epochs, test set) ---",
        f"FP32 baseline accuracy              : {fmt(qat['fp32_baseline_accuracy'])}  (Macro F1 {fmt(qat.get('fp32_baseline_macro_f1'))})",
        f"Naive fake-quant, no fine-tuning     : {fmt(qat['naive_fake_quant_accuracy'])}  (Macro F1 {fmt(qat.get('naive_fake_quant_macro_f1'))})",
        f"Fake-quant AFTER QAT fine-tuning     : {fmt(qat['qat_finetuned_accuracy'])}  (Macro F1 {fmt(qat.get('qat_finetuned_macro_f1'))})",
        "",
        "Note: naive fake-quantization changed accuracy by less than 0.1 point",
        "relative to FP32, so there was no quantization-induced collapse to",
        "recover from. The gain after QAT fine-tuning comes from additional",
        "epochs of fusion-head training (the FP32 checkpoint was chosen at",
        "epoch 9 of 10 by validation accuracy) and is not evidence that QAT",
        "itself improves accuracy. Fake-quantization is also only applied to",
        "the fusion head's input features, not real INT8 inference.",
        "",
    ]
    with open(os.path.join(OUTPUT_DIR, "results.txt"), "w") as f:
        f.write("\n".join(results_txt_lines))
    print(f"Wrote {OUTPUT_DIR}/results.txt")

    # ---------------- evaluation_results.txt ----------------
    width = max(len(n) for n in class_names) + 2
    lines = []
    lines.append("ProposedTrafficCLIP TEST RESULTS")
    lines.append("=" * 70)
    lines.append(f"Number of classes: {num_classes}")
    lines.append(f"Test samples: {test_total}")
    lines.append(f"Accuracy: {accuracy:.4f}")
    lines.append(f"Macro Precision: {macro_precision:.4f}")
    lines.append(f"Macro Recall: {macro_recall:.4f}")
    lines.append(f"Macro F1: {macro_f1:.4f}")
    lines.append("")
    lines.append("Classification Report")
    lines.append("=" * 70)
    lines.append(f"{'':>{width}}{'precision':>12}{'recall':>12}{'f1-score':>12}{'support':>12}")
    lines.append("")
    for name, p, r, f_, s in zip(class_names, precision, recall, f1, support):
        lines.append(f"{name:>{width}}{p:>12.2f}{r:>12.2f}{f_:>12.2f}{int(s):>12}")
    lines.append("")
    lines.append(f"{'accuracy':>{width}}{'':>12}{'':>12}{accuracy:>12.2f}{test_total:>12}")
    lines.append(f"{'macro avg':>{width}}{macro_precision:>12.2f}{macro_recall:>12.2f}{macro_f1:>12.2f}{test_total:>12}")
    lines.append("")
    lines.append("Confusion Matrix")
    lines.append("=" * 70)
    lines.append("Rows = true class, Columns = predicted class, order matches list above")
    lines.append(f"Classes: {class_names}")
    for row in cm:
        lines.append("[" + " ".join(f"{v:3d}" for v in row) + "]")
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
    lines.append(f"FP32 baseline accuracy              : {fmt(qat['fp32_baseline_accuracy'])}")
    lines.append(f"Naive fake-quant, no fine-tuning     : {fmt(qat['naive_fake_quant_accuracy'])}")
    lines.append(f"Fake-quant AFTER QAT fine-tuning     : {fmt(qat['qat_finetuned_accuracy'])}")
    lines.append(f"Macro F1 (FP32 / naive / QAT)        : {fmt(qat.get('fp32_baseline_macro_f1'))} / "
                 f"{fmt(qat.get('naive_fake_quant_macro_f1'))} / {fmt(qat.get('qat_finetuned_macro_f1'))}")

    with open(os.path.join(OUTPUT_DIR, "evaluation_results.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUTPUT_DIR}/evaluation_results.txt")

    # ---------------- results/figures + results/metrics ----------------
    figures_dir = os.path.join(RESULTS_DIR, "figures")
    metrics_dir = os.path.join(RESULTS_DIR, "metrics")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    # Clear old copies first so no stale figure from an earlier run survives.
    for old in os.listdir(figures_dir):
        if old.endswith(".png"):
            os.remove(os.path.join(figures_dir, old))
    for fname in FIGURES:
        src = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(figures_dir, fname))

    with open(os.path.join(metrics_dir, "per_class_results.csv"), "w") as f:
        f.write("class,precision,recall,f1,support\n")
        for name, p, r, f_, s in zip(class_names, precision, recall, f1, support):
            f.write(f"{name},{p:.4f},{r:.4f},{f_:.4f},{int(s)}\n")

    def csv_val(v):
        return "" if v is None else f"{v:.4f}"

    with open(os.path.join(metrics_dir, "summary_results.csv"), "w") as f:
        f.write("metric,value\n")
        f.write(f"num_classes,{num_classes}\n")
        f.write(f"closed_set_accuracy,{accuracy:.4f}\n")
        f.write(f"closed_set_macro_f1,{macro_f1:.4f}\n")
        f.write(f"closed_set_macro_precision,{macro_precision:.4f}\n")
        f.write(f"closed_set_macro_recall,{macro_recall:.4f}\n")
        if open_set:
            f.write(f"open_set_roc_auc,{open_set['auc']:.4f}\n")
        f.write(f"qat_fp32_baseline_accuracy,{csv_val(qat['fp32_baseline_accuracy'])}\n")
        f.write(f"qat_naive_fake_quant_accuracy,{csv_val(qat['naive_fake_quant_accuracy'])}\n")
        f.write(f"qat_finetuned_accuracy,{csv_val(qat['qat_finetuned_accuracy'])}\n")
        f.write(f"qat_fp32_baseline_macro_f1,{csv_val(qat.get('fp32_baseline_macro_f1'))}\n")
        f.write(f"qat_naive_fake_quant_macro_f1,{csv_val(qat.get('naive_fake_quant_macro_f1'))}\n")
        f.write(f"qat_finetuned_macro_f1,{csv_val(qat.get('qat_finetuned_macro_f1'))}\n")

    print(f"Wrote {metrics_dir}/per_class_results.csv")
    print(f"Wrote {metrics_dir}/summary_results.csv")
    print(f"Copied figures to {figures_dir}/")


if __name__ == "__main__":
    main()
