#!/usr/bin/env python
"""Train the 24-class AMC CNN on RadioML 2018 dataset.

Usage:
    python -m rf_analyzer.amc.train_amc
    python -m rf_analyzer.amc.train_amc --epochs 20 --max-per-class 5000

This script:
  1. Loads a subset of RadioML 2018 (filtered by SNR >= 0 dB)
  2. Trains a VT-CNN2 model for 24-class modulation classification
  3. Saves weights to rf_analyzer/amc/weights/amc_radioml2018.pt
  4. Prints per-class accuracy and accuracy-vs-SNR summary
"""
import argparse
import os
import sys
import time
import numpy as np

# Ensure the package is importable when run as a script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rf_analyzer.amc.radioml_loader import (
    load_dataset_for_training,
    accuracy_vs_snr,
    RADIOML_CLASSES,
    NUM_CLASSES,
)
from rf_analyzer.amc.cnn_model import (
    VTCNN2,
    train_model,
    load_model,
    DEFAULT_WEIGHTS_PATH,
    classify_iq_cnn,
)


def main():
    parser = argparse.ArgumentParser(description="Train AMC CNN on RadioML 2018")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--min-snr", type=float, default=0.0,
                        help="Minimum SNR for training data (dB)")
    parser.add_argument("--max-per-class", type=int, default=4000,
                        help="Max samples per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, auto-detect if omitted)")
    parser.add_argument("--save-path", type=str, default=str(DEFAULT_WEIGHTS_PATH),
                        help="Path to save model weights")
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Override dataset HDF5 path")
    args = parser.parse_args()

    print("=" * 60)
    print("  RadioML 2018 AMC Training")
    print("=" * 60)

    t0 = time.time()

    # Load dataset
    data = load_dataset_for_training(
        min_snr=args.min_snr,
        max_per_class=args.max_per_class,
        dataset_path=args.dataset_path,
        seed=args.seed,
    )

    print(f"\n[Data] Load time: {time.time() - t0:.1f}s")
    print(f"[Data] Train shape: {data['X_train'].shape}")
    print(f"[Data] Val shape:   {data['X_val'].shape}")
    print(f"[Data] Test shape:  {data['X_test'].shape}")

    # Check class distribution
    print("\n[Data] Class distribution (train):")
    for ci in range(NUM_CLASSES):
        count = np.sum(data["y_train"] == ci)
        print(f"  {RADIOML_CLASSES[ci]:>10s}: {count:5d}")

    # Train
    t1 = time.time()
    model, history = train_model(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_val=data["X_val"],
        y_val=data["y_val"],
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path,
        device=args.device,
    )
    print(f"\n[AMC] Training time: {time.time() - t1:.1f}s")

    # Evaluate on test set
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    X_test = torch.FloatTensor(data["X_test"]).to(device)
    y_test = data["y_test"]

    # Predict in batches
    all_preds = []
    for i in range(0, len(X_test), args.batch_size):
        batch = X_test[i : i + args.batch_size]
        with torch.no_grad():
            preds = model(batch).argmax(1).cpu().numpy()
        all_preds.append(preds)
    y_pred = np.concatenate(all_preds)

    overall_acc = float(np.mean(y_pred == y_test))
    print(f"\n{'=' * 60}")
    print(f"  Test Results")
    print(f"{'=' * 60}")
    print(f"  Overall accuracy: {overall_acc:.4f} ({overall_acc*100:.1f}%)")

    # Per-class accuracy
    print(f"\n  Per-class accuracy:")
    print(f"  {'Class':>10s}  {'Correct':>7s}  {'Total':>5s}  {'Accuracy':>8s}")
    print(f"  {'-'*36}")
    for ci in range(NUM_CLASSES):
        mask = y_test == ci
        if mask.sum() > 0:
            class_acc = float(np.mean(y_pred[mask] == y_test[mask]))
            print(f"  {RADIOML_CLASSES[ci]:>10s}  {int(np.sum(y_pred[mask] == ci)):>7d}  "
                  f"{int(mask.sum()):>5d}  {class_acc:>8.4f}")

    # Accuracy vs SNR (using a quick subset)
    print(f"\n  Accuracy vs SNR (on dataset, 100 samples/class/SNR):")

    def predict_fn(X):
        batch_t = torch.FloatTensor(X).to(device)
        with torch.no_grad():
            return model(batch_t).argmax(1).cpu().numpy()

    snr_results = accuracy_vs_snr(
        predict_fn,
        max_per_class_per_snr=100,
        dataset_path=args.dataset_path,
    )

    print(f"  {'SNR (dB)':>8s}  {'Accuracy':>8s}")
    print(f"  {'-'*20}")
    for snr_db in sorted(snr_results.keys()):
        print(f"  {snr_db:>8.0f}  {snr_results[snr_db]:>8.4f}")

    print(f"\n[AMC] Model saved to: {args.save_path}")
    print(f"[AMC] Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
