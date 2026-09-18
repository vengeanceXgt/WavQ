#!/usr/bin/env python
"""Evaluate the full RF analysis pipeline against RadioML 2018 dataset.

Usage:
    python -m rf_analyzer.eval.evaluate_radioml
    python -m rf_analyzer.eval.evaluate_radioml --samples-per-combo 20 --output results.json

This script:
  1. Loads representative samples from RadioML 2018
  2. Runs CNN-based AMC classification
  3. Runs the full pipeline on selected samples
  4. Generates per-class, per-SNR accuracy reports
  5. Saves results as JSON
"""
import argparse
import json
import os
import sys
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rf_analyzer.amc.radioml_loader import (
    load_index_arrays,
    get_indices_for_subset,
    load_samples,
    RADIOML_CLASSES,
    NUM_CLASSES,
)
from rf_analyzer.amc.cnn_model import classify_iq_cnn, load_model
from rf_analyzer.orchestrator.pipeline import run_pipeline_on_iq


def evaluate_amc_accuracy(
    max_per_class_per_snr: int = 50,
    target_snrs=None,
    target_mods=None,
    dataset_path=None,
):
    """Evaluate AMC CNN accuracy across modulations and SNR levels.

    Returns:
        dict with per-mod, per-SNR accuracy and overall summary
    """
    print("[Eval] Loading dataset index arrays...")
    labels, snrs = load_index_arrays(dataset_path)
    unique_snrs = sorted(np.unique(snrs))

    if target_snrs is not None:
        unique_snrs = [s for s in unique_snrs if s in target_snrs]
    if target_mods is None:
        target_mods = RADIOML_CLASSES

    model = load_model()
    if model is None:
        print("[Eval] ERROR: No trained CNN model found. Run train_amc.py first.")
        return None

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    results = []
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

    for snr_val in unique_snrs:
        print(f"\n[Eval] SNR = {snr_val:.0f} dB")
        for mod_name in target_mods:
            mod_idx = RADIOML_CLASSES.index(mod_name)
            indices = get_indices_for_subset(
                labels, snrs,
                modulations=[mod_name],
                min_snr=snr_val, max_snr=snr_val,
                max_per_class=max_per_class_per_snr,
            )
            if len(indices) == 0:
                continue

            # Load as complex for CNN classification
            X_complex = load_samples(indices, dataset_path, as_complex=True)

            correct = 0
            for i in range(len(X_complex)):
                result = classify_iq_cnn(X_complex[i], model=model, device=device)
                pred_mod = result["modulation"]

                # Map prediction to class index
                pred_mod_upper = pred_mod.upper() if pred_mod else ""
                if pred_mod_upper in RADIOML_CLASSES:
                    pred_idx = RADIOML_CLASSES.index(pred_mod_upper)
                else:
                    pred_idx = -1

                if pred_idx >= 0:
                    confusion[mod_idx, pred_idx] += 1

                if pred_mod_upper == mod_name:
                    correct += 1

            acc = correct / len(X_complex)
            results.append({
                "modulation": mod_name,
                "snr_db": float(snr_val),
                "accuracy": acc,
                "n_samples": len(X_complex),
                "correct": correct,
            })
            print(f"  {mod_name:>10s}: {acc:.3f} ({correct}/{len(X_complex)})")

    return {
        "per_result": results,
        "confusion_matrix": confusion.tolist(),
        "classes": RADIOML_CLASSES,
    }


def evaluate_full_pipeline(
    n_samples_per_mod: int = 5,
    target_snr: float = 18.0,
    target_mods=None,
    dataset_path=None,
):
    """Run a few samples through the FULL pipeline (not just AMC).

    This tests whether the pipeline runs end-to-end without crashes
    and reports reasonable stage results.
    """
    print(f"\n[Eval] Full pipeline test at SNR={target_snr} dB")
    labels, snrs = load_index_arrays(dataset_path)

    if target_mods is None:
        target_mods = ["BPSK", "QPSK", "16QAM", "FM", "OOK"]

    pipeline_results = []

    for mod_name in target_mods:
        indices = get_indices_for_subset(
            labels, snrs,
            modulations=[mod_name],
            min_snr=target_snr, max_snr=target_snr,
            max_per_class=n_samples_per_mod,
        )
        if len(indices) == 0:
            print(f"  {mod_name}: no samples at SNR={target_snr}")
            continue

        X_complex = load_samples(indices, dataset_path, as_complex=True)

        for i in range(len(X_complex)):
            try:
                stages = run_pipeline_on_iq(
                    X_complex[i],
                    sample_rate=1.0,
                    use_cnn_amc=True,
                )
                stage_names = [s.name for s in stages]
                stage_ok = [s.ok for s in stages]
                amc_stage = next((s for s in stages if s.name == "amc"), None)
                detected = None
                if amc_stage and amc_stage.ok and isinstance(amc_stage.data, dict):
                    detected = amc_stage.data.get("modulation")

                pipeline_results.append({
                    "true_mod": mod_name,
                    "detected_mod": detected,
                    "stages_reached": len(stages),
                    "stages_ok": sum(stage_ok),
                    "stage_names": stage_names,
                    "no_crash": True,
                })
            except Exception as e:
                pipeline_results.append({
                    "true_mod": mod_name,
                    "detected_mod": None,
                    "stages_reached": 0,
                    "no_crash": False,
                    "error": str(e),
                })

        n_ok = sum(1 for r in pipeline_results if r["true_mod"] == mod_name and r["no_crash"])
        print(f"  {mod_name:>10s}: {n_ok}/{len(indices)} completed without crash")

    return pipeline_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate pipeline against RadioML 2018")
    parser.add_argument("--samples-per-combo", type=int, default=30,
                        help="Samples per modulation/SNR combo for AMC eval")
    parser.add_argument("--pipeline-samples", type=int, default=3,
                        help="Samples per mod for full pipeline test")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--snrs", type=str, default="-8,-4,0,4,8,12,16,20,24,28",
                        help="Comma-separated SNR values to test")
    args = parser.parse_args()

    target_snrs = [float(s) for s in args.snrs.split(",")]

    print("=" * 60)
    print("  RadioML 2018 Pipeline Evaluation")
    print("=" * 60)

    t0 = time.time()

    # AMC accuracy evaluation
    amc_results = evaluate_amc_accuracy(
        max_per_class_per_snr=args.samples_per_combo,
        target_snrs=target_snrs,
        dataset_path=args.dataset_path,
    )

    if amc_results:
        # Print summary table
        print(f"\n{'='*60}")
        print("  AMC Accuracy Summary (by SNR)")
        print(f"{'='*60}")

        snr_acc = defaultdict(list)
        for r in amc_results["per_result"]:
            snr_acc[r["snr_db"]].append(r["accuracy"])

        print(f"  {'SNR (dB)':>8s}  {'Avg Accuracy':>12s}  {'N Classes':>9s}")
        print(f"  {'-'*35}")
        for snr_db in sorted(snr_acc.keys()):
            avg = np.mean(snr_acc[snr_db])
            print(f"  {snr_db:>8.0f}  {avg:>12.4f}  {len(snr_acc[snr_db]):>9d}")

    # Full pipeline evaluation
    pipeline_results = evaluate_full_pipeline(
        n_samples_per_mod=args.pipeline_samples,
        target_snr=18.0,
        dataset_path=args.dataset_path,
    )

    # Save results
    output = {
        "amc_results": amc_results,
        "pipeline_results": pipeline_results,
        "elapsed_seconds": time.time() - t0,
    }

    out_path = args.output
    if out_path is None:
        out_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "radioml2018_eval.json")

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[Eval] Results saved to: {out_path}")
    print(f"[Eval] Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
