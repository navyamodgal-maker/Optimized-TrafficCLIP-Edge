# Proposed TrafficCLIP — Behavioral Dynamics Modality

Tri-modal (image + text + behavioral-statistics) traffic classifier,
built on TrafficCLIP / OptimizedTrafficCLIP, extended with:

1. **Behavioral Dynamics Modality** (renamed from "Physics-Aware"):
   8 session-level statistics (mean IAT, IAT variance, jitter, byte
   entropy, mean packet length, packet-length variance, flow
   duration, burstiness) capture protocol-level behavioral dynamics —
   e.g. TCP congestion control directly shapes inter-arrival time
   (IAT) distributions, and bursty vs. steady-state transmission
   shows up in the burstiness and jitter statistics.
2. **Quantization-Aware Training (QAT)** for the fusion head, to fix
   the post-training-quantization (PTQ) accuracy collapse reported
   in the OptimizedTrafficCLIP paper.
3. **Open-set / zero-day detection**: two classes are held out
   entirely from training and evaluated as "unknown" traffic at test
   time, with an ROC-AUC score for known-vs-unknown separation.

Dataset: USTC-TFC2016 (10 classes), pulled from a public GitHub
mirror — no signup required.

## Run order (on Kaggle, with GPU enabled)

```bash
pip install -r requirements.txt --quiet

# 1. Download raw pcaps for all 10 classes
python scripts/01_download_pcap_data.py

# 2. Build the tri-modal dataset (images + 8 stats -> npz)
#    --max-flows-per-class keeps this fast; raise it if you have time.
python scripts/02_build_dataset.py --max-flows-per-class 300

# 3. Train the standard (closed-set) model
python -m src.train_proposed

# 4. Train the open-set model (Zeus + BitTorrent held out)
python -m src.train_proposed --open-set

# 5. Evaluate: standard metrics + open-set ROC-AUC
python -m src.evaluate_proposed --checkpoint output/proposed_traffic_clip_open_set_best.pth

# 6. Quantization-aware training on the fusion head
python -m src.qat_train --checkpoint output/proposed_traffic_clip_best.pth
```

## What to report from tonight's run

- **Closed-set accuracy/F1** from step 3 (this is your reproducible,
  presentable headline number).
- **ROC-AUC for open-set detection** from step 5 — this is your
  answer to "can the model recognize zero-day/unknown traffic".
- **FP32 vs. naive-fake-quant vs. QAT-fine-tuned accuracy** from
  step 6 — this directly answers the paper's INT8-quantization
  weakness; report the gap QAT closes vs. no fine-tuning at all.

## Notes / honest limitations to mention when presenting

- Trained on a subset of flows per class (`--max-flows-per-class`)
  rather than the full USTC-TFC2016 corpus, due to a same-week
  deadline — full-corpus training is the natural next step.
- QAT here targets the fusion head specifically (the actual
  trainable, deployable classifier); the ResNet-50 and BERT
  backbones are frozen regardless of quantization, so they were not
  included in the quantization scope.
- Full INT8 ONNX export (with QDQ nodes) and edge-hardware
  benchmarking (Raspberry Pi / Jetson) were out of scope for this
  iteration.
