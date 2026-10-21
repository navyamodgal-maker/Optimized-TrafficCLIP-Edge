# Optimized TrafficCLIP — USTC-TFC2016 20-Class Evaluation

Tri-modal (traffic image + text prompt + 8 behavioural statistics) network-traffic
classification, extending the TrafficCLIP / OptimizedTrafficCLIP codebase from 10 to
**20 USTC-TFC2016 classes**.

Classes: `Skype`, `MySQL`, `BitTorrent`, `Facetime`, `Weibo`, `Zeus`, `Tinba`, `Cridex`,
`Geodo`, `Miuref`, `FTP`, `Gmail`, `Outlook`, `SMB`, `WorldOfWarcraft`, `Htbot`, `Neris`,
`Nsis-ay`, `Shifu`, `Virut`.

## Setup used for the reported results

- 500 flows per class (10,000 total), stratified 70 / 15 / 15 split, seed 42
  → 1,500 test samples (75 per class)
- 10 epochs, batch size 16, AdamW (lr 1e-4); run on Kaggle
- 8 statistics: mean IAT, IAT variance, jitter, byte entropy, mean packet length,
  packet-length variance, flow duration, burstiness

## Results (test set, 20 classes)

| Metric | Value |
|---|---:|
| Accuracy | 0.8220 |
| Macro precision | 0.8482 |
| Macro recall | 0.8220 |
| Macro F1 | 0.8162 |
| Open-set ROC-AUC (Zeus, BitTorrent held out) | 0.7712 |
| Best validation accuracy, closed-set | 0.8367 |
| Best validation accuracy, open-set (18 known classes) | 0.8563 |

Weakest classes (F1): Miuref 0.50 (half of its samples are predicted as Geodo),
Virut 0.56 (33 of 75 predicted as Neris), BitTorrent 0.65 (17 of 75 → Outlook;
23 Outlook samples → BitTorrent), Neris 0.69, FTP 0.70, Gmail 0.70, Outlook 0.70.

### Quantization-aware training (fusion head only)

| Setting | Test accuracy |
|---|---:|
| FP32 baseline | 0.8220 |
| Fake-quantized, no QAT training | 0.8213 |
| Fake-quantized, after 5 QAT epochs | 0.8573 |

How to read this:
- Naive fake-quantization costs < 0.1 point, so there was no quantization collapse to repair.
- The +3.5 point gain comes from 5 extra epochs of fusion-head training on top of an
  epoch-9 checkpoint; without an FP32-continued-training control it cannot be attributed
  to QAT.
- Only the fusion head's input features are fake-quantized (not its weights), and no real
  INT8 inference is run.
- These are **accuracy** values. Macro F1 for the QAT runs was not recorded; re-running
  `python -m src.qat_train` now computes and saves it to `output/qat_results.json`.

## Figures

`output/` (copied to `results/figures/`): `confusion_matrix.png`,
`confusion_matrix_normalized.png`, `per_class_metrics.png`, `open_set_roc_curve.png`,
`training_curves.png`, `qat_comparison.png`.

## Reproducing

```bash
pip install -r requirements.txt
python scripts/01_download_pcap_data.py
python scripts/02_build_dataset.py --max-flows-per-class 500
python -m src.train_proposed                 # closed-set, 20 classes
python -m src.train_proposed --open-set      # Zeus + BitTorrent held out
python -m src.evaluate_proposed              # writes results_summary.json + figures
python -m src.qat_train                      # writes qat_results.json
python scripts/04_generate_figures.py        # redraw figures from saved JSON (no GPU)
python scripts/03_generate_report_outputs.py # results.txt, evaluation_results.txt, CSVs
```

## Known limitations

- Open-set AUC is computed over the whole dataset, so known-class samples include ones the
  open-set model was trained on; this likely makes the 0.7712 optimistic. Evaluating
  known classes on the test split only would be cleaner.
- `output/training_logs.json` and the QAT accuracies were transcribed from the Kaggle console;
  they are not regenerated automatically.
- Validation accuracy was still rising slowly at epoch 10; longer training may help.
