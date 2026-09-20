"""MVP Configuration."""

CONFIG = {
    "processing": {
        "max_file_size_mb": 500,
        "default_sample_rate_hz": 1.0,
        "chunk_size_samples": 1024 * 1024
    },
    "spectral": {
        "fft_size": 2048,
        "psd_averaging_blocks": 10,
        "cfo_power_law": 4,
        "occupied_bw_threshold_db": -20.0
    },
    "synchronization": {
        "timing_sps_min": 2,
        "timing_sps_max": 32,
        "gardner_loop_bw": 0.01,
        "costas_loop_bw": 0.05,
        "costas_damping": 0.707
    },
    "modulation": {
        "supported_classes": ["2fsk", "bpsk", "qpsk", "16qam"],
        "ambiguity_threshold_ll": 10.0,
        "envelope_tolerance": 0.15,
        "cumulant_fusion_weight": 50.0
    },
    "fec": {
        "supported_schemes": ["viterbi_k7", "reed_solomon"],
        "interleaver_max_width": 256
    },
    "export": {
        "generate_plots": True,
        "generate_report": True,
        "max_plot_samples": 2000
    }
}
