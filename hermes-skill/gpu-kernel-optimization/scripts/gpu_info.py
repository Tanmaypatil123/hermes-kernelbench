#!/usr/bin/env python3
"""gpu_info.py - print the numbers a kernel design depends on, for the GPU this runs on.

  python3 gpu_info.py            # human-readable
  python3 gpu_info.py --json     # one JSON object

Reads the device through PyTorch (SM count, L2 size, shared memory limits, compute capability) and
looks up peak DRAM bandwidth by name, since drivers do not report it. Run it where the kernel runs:
locally, or inside the task's remote test.
"""
import argparse, json, sys

# Peak DRAM bandwidth in GB/s, matched by substring of the device name (first match wins).
PEAK_GBPS = [
    ("H200", 4800), ("H100 PCIe", 2000), ("H100", 3350), ("B200", 8000), ("GB200", 8000),
    ("A100-SXM4-80GB", 2039), ("A100 80GB", 2039), ("A100", 1555),
    ("RTX PRO 6000", 1792), ("5090", 1792), ("4090", 1008), ("3090", 936),
    ("L40S", 864), ("L40", 864), ("L4", 300), ("A10G", 600), ("A10", 600), ("A40", 696),
    ("T4", 320), ("V100", 900),
]
ARCH = {(7, 0): "Volta", (7, 5): "Turing", (8, 0): "Ampere", (8, 6): "Ampere", (8, 7): "Ampere",
        (8, 9): "Ada", (9, 0): "Hopper", (10, 0): "Blackwell (datacenter)",
        (12, 0): "Blackwell (consumer)"}


def info():
    try:
        import torch
    except ImportError:
        sys.exit("gpu_info: PyTorch is not installed here")
    if not torch.cuda.is_available():
        sys.exit("gpu_info: no CUDA device visible from this machine; run it where the kernel runs")
    p = torch.cuda.get_device_properties(0)
    cc = (p.major, p.minor)
    peak = next((bw for key, bw in PEAK_GBPS if key in p.name), None)
    d = {
        "name": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "arch": ARCH.get(cc, "unknown"),
        "nvcc_gencode": f"-gencode=arch=compute_{p.major}{p.minor},code=sm_{p.major}{p.minor}",
        "sm_count": p.multi_processor_count,
        "l2_cache_mb": round(getattr(p, "L2_cache_size", 0) / 2**20, 1) or None,
        "shared_mem_per_block_optin_kb": round(getattr(p, "shared_memory_per_block_optin", 0) / 1024) or None,
        "shared_mem_per_sm_kb": round(getattr(p, "shared_memory_per_multiprocessor", 0) / 1024) or None,
        "regs_per_sm": getattr(p, "regs_per_multiprocessor", None),
        "max_threads_per_sm": getattr(p, "max_threads_per_multi_processor", None),
        "memory_gb": round(p.total_memory / 2**30, 1),
        "peak_dram_gbps": peak,
        "bf16_tensor_cores": cc >= (8, 0),
        "fp8_tensor_cores": cc >= (8, 9),
        "wgmma_tma": cc == (9, 0),
        "tcgen05_tmem": cc[0] == 10,
        "torch": torch.__version__,
    }
    try:
        import triton
        d["triton"] = triton.__version__
    except ImportError:
        d["triton"] = None
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", action="store_true")
    d = info()
    if ap.parse_args().json:
        print(json.dumps(d))
        return
    for k, v in d.items():
        print(f"{k:<32} {v}")
    if d["peak_dram_gbps"]:
        print(f"\nroofline: a kernel moving B bytes cannot finish faster than B / {d['peak_dram_gbps']}e9 s;"
              f" 1 GB takes at least {1000 / d['peak_dram_gbps']:.2f} ms here.")
    else:
        print("\npeak bandwidth unknown for this GPU name; look it up and compute the roofline by hand.")


if __name__ == "__main__":
    main()
