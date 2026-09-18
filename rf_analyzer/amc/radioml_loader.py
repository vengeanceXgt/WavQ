"""RadioML 2018 (GOLD_XYZ_OSC.0001_1024) dataset loader.

Handles loading and preprocessing the RadioML 2018.01A dataset from HDF5.
The dataset contains 2,555,904 IQ recordings across 24 modulation classes
at SNR levels from -20 dB to +30 dB (2 dB steps).

Dataset structure in HDF5:
    X: (2555904, 1024, 2) -- I/Q time-series, 1024 samples per recording
    Y: (2555904, 24)      -- one-hot encoded modulation labels
    Z: (2555904, 1)       -- SNR in dB

Memory strategy: the full dataset is ~21 GB. We never load it all into RAM.
Instead, we precompute indices for the desired subset (by SNR, modulation,
or random subsample) and load only those rows from the HDF5 file.
"""
import os
import numpy as np
import h5py
from typing import Optional, Tuple, Dict, List

# 24 modulation classes in the order stored in the dataset's one-hot encoding.
# This ordering matches classes-fixed.json in the dataset distribution.
RADIOML_CLASSES = [
    "OOK", "4ASK", "8ASK",
    "BPSK", "QPSK", "8PSK",
    "16PSK", "32PSK", "16APSK",
    "32APSK", "64APSK", "128APSK",
    "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC",
    "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]

NUM_CLASSES = len(RADIOML_CLASSES)

# Default dataset location
DEFAULT_DATASET_PATH = (
    r"D:\Aukaat\dataset_cache\datasets\pinxau1000\radioml2018"
    r"\versions\2\GOLD_XYZ_OSC.0001_1024.hdf5"
)


def _find_dataset_path() -> str:
    """Locate the dataset HDF5 file."""
    if os.path.exists(DEFAULT_DATASET_PATH):
        return DEFAULT_DATASET_PATH
    # Check environment variable override
    env_path = os.environ.get("RADIOML_DATASET_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    raise FileNotFoundError(
        f"RadioML 2018 dataset not found at: {DEFAULT_DATASET_PATH}\n"
        f"Set RADIOML_DATASET_PATH environment variable to override."
    )


def load_index_arrays(
    dataset_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load only Y (labels) and Z (SNR) arrays — computed in chunks to avoid OOM.

    The Y array is (2555904, 24) stored as int64 in HDF5, which is ~460 MB
    if loaded at once. Instead, we process it in chunks and extract only
    the argmax (class index) per row, keeping memory under 30 MB.

    Returns:
        labels: (N,) int32 class indices (0..23)
        snrs:   (N,) float32 SNR values in dB
    """
    path = dataset_path or _find_dataset_path()
    CHUNK = 100000

    with h5py.File(path, "r") as f:
        n_total = f["Y"].shape[0]

        # Pre-allocate output arrays
        labels = np.empty(n_total, dtype=np.int32)
        snrs = np.empty(n_total, dtype=np.float32)

        for start in range(0, n_total, CHUNK):
            end = min(start + CHUNK, n_total)
            y_chunk = f["Y"][start:end]   # (chunk, 24) — ~18 MB max
            labels[start:end] = np.argmax(y_chunk, axis=1).astype(np.int32)
            snrs[start:end] = f["Z"][start:end].ravel().astype(np.float32)

    return labels, snrs


def get_indices_for_subset(
    labels: np.ndarray,
    snrs: np.ndarray,
    modulations: Optional[List[str]] = None,
    min_snr: Optional[float] = None,
    max_snr: Optional[float] = None,
    max_per_class: Optional[int] = None,
    seed: int = 42,
) -> np.ndarray:
    """Select a subset of dataset indices based on modulation, SNR, and count.

    Args:
        labels: integer class indices array
        snrs: SNR values array
        modulations: list of modulation names to include (None = all 24)
        min_snr: minimum SNR (inclusive)
        max_snr: maximum SNR (inclusive)
        max_per_class: max samples per class (subsamples if larger)
        seed: random seed for reproducible subsampling

    Returns:
        sorted array of selected indices
    """
    mask = np.ones(len(labels), dtype=bool)

    if modulations is not None:
        class_indices = [RADIOML_CLASSES.index(m) for m in modulations]
        mod_mask = np.zeros(len(labels), dtype=bool)
        for ci in class_indices:
            mod_mask |= (labels == ci)
        mask &= mod_mask

    if min_snr is not None:
        mask &= (snrs >= min_snr)
    if max_snr is not None:
        mask &= (snrs <= max_snr)

    indices = np.where(mask)[0]

    if max_per_class is not None and max_per_class > 0:
        rng = np.random.default_rng(seed)
        selected = []
        for ci in range(NUM_CLASSES):
            cls_idx = indices[labels[indices] == ci]
            if len(cls_idx) > max_per_class:
                cls_idx = rng.choice(cls_idx, size=max_per_class, replace=False)
            selected.append(cls_idx)
        indices = np.sort(np.concatenate(selected))

    return indices


def load_samples(
    indices: np.ndarray,
    dataset_path: Optional[str] = None,
    as_complex: bool = False,
) -> np.ndarray:
    """Load IQ samples for given indices from HDF5.

    For CNN training: returns (N, 2, 1024) float32 — I and Q as channels.
    For pipeline use: returns (N,) array of complex64 1024-length vectors.

    Args:
        indices: sorted array of row indices to load
        dataset_path: path to HDF5 file
        as_complex: if True, return complex64; if False, return (N, 2, 1024)

    Returns:
        IQ data array
    """
    path = dataset_path or _find_dataset_path()

    # HDF5 fancy indexing requires sorted indices for performance.
    # Load in chunks to avoid OOM on huge selections.
    CHUNK = 10000
    results = []
    with h5py.File(path, "r") as f:
        X = f["X"]
        for start in range(0, len(indices), CHUNK):
            chunk_idx = indices[start : start + CHUNK]
            # X shape: (N, 1024, 2) — time x [I, Q]
            data = X[chunk_idx]  # (chunk, 1024, 2) float32
            results.append(data)

    data = np.concatenate(results, axis=0)  # (N, 1024, 2)

    if as_complex:
        # Convert to complex64: I + jQ
        return (data[:, :, 0] + 1j * data[:, :, 1]).astype(np.complex64)
    else:
        # Transpose to (N, 2, 1024) for CNN input format: [channels, time]
        return data.transpose(0, 2, 1).astype(np.float32)


def train_val_test_split(
    indices: np.ndarray,
    labels: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split of indices.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []

    for ci in range(NUM_CLASSES):
        cls_mask = labels[indices] == ci
        cls_indices = indices[cls_mask]
        rng.shuffle(cls_indices)

        n = len(cls_indices)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train_idx.append(cls_indices[:n_train])
        val_idx.append(cls_indices[n_train : n_train + n_val])
        test_idx.append(cls_indices[n_train + n_val :])

    return (
        np.sort(np.concatenate(train_idx)),
        np.sort(np.concatenate(val_idx)),
        np.sort(np.concatenate(test_idx)),
    )


def load_dataset_for_training(
    min_snr: float = 0.0,
    max_per_class: int = 4000,
    dataset_path: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """High-level convenience: load a training-ready subset.

    Returns dict with keys: X_train, y_train, X_val, y_val, X_test, y_test
    All X arrays are shape (N, 2, 1024), y arrays are (N,) int64.
    """
    path = dataset_path or _find_dataset_path()
    print(f"[RadioML] Loading index arrays from {path}...")
    labels, snrs = load_index_arrays(path)

    print(f"[RadioML] Total samples in dataset: {len(labels)}")
    print(f"[RadioML] Filtering: SNR >= {min_snr} dB, max {max_per_class}/class")

    indices = get_indices_for_subset(
        labels, snrs,
        min_snr=min_snr,
        max_per_class=max_per_class,
        seed=seed,
    )
    print(f"[RadioML] Selected {len(indices)} samples")

    train_idx, val_idx, test_idx = train_val_test_split(
        indices, labels, seed=seed
    )
    print(f"[RadioML] Splits — Train: {len(train_idx)}, "
          f"Val: {len(val_idx)}, Test: {len(test_idx)}")

    X_train = load_samples(train_idx, path)
    y_train = labels[train_idx]
    X_val = load_samples(val_idx, path)
    y_val = labels[val_idx]
    X_test = load_samples(test_idx, path)
    y_test = labels[test_idx]

    # Per-sample energy normalization
    for X in (X_train, X_val, X_test):
        energy = np.sqrt(np.mean(X ** 2, axis=(1, 2), keepdims=True))
        energy = np.maximum(energy, 1e-10)
        X /= energy

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }


def accuracy_vs_snr(
    predict_fn,
    modulations: Optional[List[str]] = None,
    max_per_class_per_snr: int = 200,
    dataset_path: Optional[str] = None,
    seed: int = 42,
) -> Dict[float, float]:
    """Compute classification accuracy at each SNR level.

    Args:
        predict_fn: callable(X) -> y_pred, where X is (N, 2, 1024) float32
        modulations: subset of modulations to evaluate (None = all)
        max_per_class_per_snr: max samples per class per SNR level
        dataset_path: path to HDF5 file

    Returns:
        dict mapping SNR_dB -> accuracy (0.0 to 1.0)
    """
    path = dataset_path or _find_dataset_path()
    labels, snrs = load_index_arrays(path)
    unique_snrs = np.unique(snrs)
    results = {}

    for snr_val in sorted(unique_snrs):
        indices = get_indices_for_subset(
            labels, snrs,
            modulations=modulations,
            min_snr=snr_val, max_snr=snr_val,
            max_per_class=max_per_class_per_snr,
            seed=seed,
        )
        if len(indices) == 0:
            continue

        X = load_samples(indices, path)
        # Normalize
        energy = np.sqrt(np.mean(X ** 2, axis=(1, 2), keepdims=True))
        X /= np.maximum(energy, 1e-10)

        y_true = labels[indices]
        y_pred = predict_fn(X)
        acc = float(np.mean(y_pred == y_true))
        results[float(snr_val)] = acc

    return results
