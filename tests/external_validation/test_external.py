import os
import glob
import json
import numpy as np
from rf_analyzer.orchestrator.pipeline import run_pipeline

def validate_external_dataset(dataset_dir):
    print(f"Validating dataset in {dataset_dir}\n")
    
    # Find all sigmf-meta files
    meta_files = glob.glob(os.path.join(dataset_dir, "**", "*.sigmf-meta"), recursive=True)
    if not meta_files:
        print("No SigMF files found.")
        return
        
    print(f"Found {len(meta_files)} recordings. Filtering subset...")
    
    # Subset Selection Rule:
    # 1 BPSK, 1 QPSK (first alphabetical match)
    selected_files = []
    for mod in ["bpsk", "qpsk", "BPSK", "QPSK"]:
        matches = [m for m in meta_files if mod.lower() in m.lower()]
        if matches:
            selected_files.append(sorted(matches)[0])
            
    # Deduplicate
    selected_files = list(set(selected_files))
    print(f"Selected {len(selected_files)} files for evaluation:")
    for f in selected_files:
        print(f" - {f}")
        
    for meta_path in selected_files:
        print(f"\n{'='*50}\nEvaluating: {os.path.basename(meta_path)}")
        data_path = meta_path.replace(".sigmf-meta", ".sigmf-data")
        
        if not os.path.exists(data_path):
            print("ERROR: Missing corresponding .sigmf-data file!")
            continue
            
        with open(meta_path) as fd:
            meta = json.load(fd)
            
        # Ground truth extraction
        capture_freq = meta.get("captures", [{}])[0].get("core:frequency")
        print(f"Ground Truth - Capture Frequency: {capture_freq}")
        
        try:
            stages = run_pipeline(data_path)
            
            last_successful = None
            first_failure = None
            
            for stage in stages:
                print(f"Stage: {stage.name:<15} | OK: {stage.ok}")
                if stage.ok:
                    last_successful = stage.name
                    if stage.name == "amc":
                        print(f"  -> Predicted Mod: {stage.data.get('modulation')}")
                    elif stage.name == "spectral":
                        print(f"  -> Absolute Freq: {stage.data.get('frequency', {}).get('absolute_rf_hz')}")
                elif first_failure is None:
                    first_failure = stage.name
                    print(f"  -> ERROR: {stage.error}")
            
            print(f"\nPipeline Summary: Last Success = {last_successful}, First Failure = {first_failure}")
        except Exception as e:
            print(f"Pipeline crashed entirely: {e}")

if __name__ == "__main__":
    validate_external_dataset("external_data/badger_kim_4603987/unzipped")
