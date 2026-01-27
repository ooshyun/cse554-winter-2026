"""
GPU Configuration Module
Auto-detects GPU count and selects appropriate device.
Use GPU 6 if 8+ GPUs detected (RTX 6000 server), otherwise GPU 0.
"""

import os
import subprocess


def detect_gpu_id() -> str:
    """Detect number of GPUs and return appropriate GPU ID."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '-L'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            num_gpus = len(result.stdout.strip().split('\n'))
            return "6" if num_gpus >= 8 else "0"
    except (FileNotFoundError, Exception):
        pass
    return "0"


def configure_gpu():
    """Configure CUDA_VISIBLE_DEVICES environment variable."""
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = detect_gpu_id()


# Auto-configure on import
configure_gpu()

# Export for convenience
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
