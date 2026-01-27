"""
GPU Specifications Constants
Multi-GPU Support: RTX 4070 Ti SUPER and Quadro RTX 6000

Sources:
- NVIDIA Ada Architecture Whitepaper v2.1
- NVIDIA Quadro RTX 6000 Datasheet
"""

import torch

# RTX 4070 Ti SUPER Specifications (Ada Lovelace Architecture)
RTX_4070_TI_SUPER_BANDWIDTH = 672.0    # GB/s (256-bit bus, 21 Gbps)
RTX_4070_TI_SUPER_COMPUTE_CAP = (8, 9)
RTX_4070_TI_SUPER_SM_COUNT = 66
RTX_4070_TI_SUPER_TFLOPS = 44.1        # FP32 TFLOPS

# Quadro RTX 6000 Specifications (Turing Architecture)
QUADRO_RTX_6000_BANDWIDTH = 672.0      # GB/s (384-bit bus, 14 Gbps)
QUADRO_RTX_6000_COMPUTE_CAP = (7, 5)
QUADRO_RTX_6000_SM_COUNT = 72
QUADRO_RTX_6000_TFLOPS = 16.3          # FP32 TFLOPS

# Default values (RTX 4070 Ti SUPER for backward compatibility)
GPU_PEAK_BANDWIDTH_DATASHEET = RTX_4070_TI_SUPER_BANDWIDTH
GPU_PEAK_TFLOPS_SPARSITY = 780.0       # FP8 with sparsity (Ada only)
GPU_MEMORY_BUS_WIDTH = 256             # bits (RTX 4070 Ti SUPER)
GPU_MEMORY_CLOCK_GBPS = 21             # Gbps (RTX 4070 Ti SUPER)


def get_gpu_peak_bandwidth():
    """
    Get peak memory bandwidth for current GPU

    Returns:
        float: Peak bandwidth in GB/s
    """
    if not torch.cuda.is_available():
        return GPU_PEAK_BANDWIDTH_DATASHEET

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    compute_cap = (props.major, props.minor)

    if compute_cap == (8, 9):
        # RTX 4070 Ti SUPER (Ada Lovelace)
        return RTX_4070_TI_SUPER_BANDWIDTH
    elif compute_cap == (7, 5):
        # Quadro RTX 6000 (Turing)
        return QUADRO_RTX_6000_BANDWIDTH
    else:
        # Fallback: calculate from device properties
        # bandwidth = 2 × memory_clock (GHz) × bus_width (bytes)
        memory_clock_ghz = props.memory_clock_rate / 1e6  # Convert kHz to GHz
        bus_width_bytes = props.memory_bus_width / 8
        return 2.0 * memory_clock_ghz * bus_width_bytes


def get_gpu_info():
    """
    Get current GPU information

    Returns:
        dict: GPU information including name, compute capability, bandwidth
    """
    if not torch.cuda.is_available():
        return {
            'name': 'No CUDA device',
            'compute_capability': (0, 0),
            'peak_bandwidth': 0.0,
            'sm_count': 0
        }

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    compute_cap = (props.major, props.minor)

    return {
        'name': props.name,
        'compute_capability': compute_cap,
        'peak_bandwidth': get_gpu_peak_bandwidth(),
        'sm_count': props.multi_processor_count,
        'memory_total': props.total_memory / (1024**3)  # GB
    }


if __name__ == '__main__':
    # Test GPU detection
    info = get_gpu_info()
    print("GPU Information:")
    print(f"  Name: {info['name']}")
    print(f"  Compute Capability: {info['compute_capability']}")
    print(f"  Peak Bandwidth: {info['peak_bandwidth']:.2f} GB/s")
    print(f"  SM Count: {info['sm_count']}")
    print(f"  Memory Total: {info['memory_total']:.2f} GB")
