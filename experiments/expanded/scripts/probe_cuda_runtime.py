"""Record the isolated runtime and run one real CUDA FP32 kernel."""
import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "reports" / "runtime_cuda_probe.json")
    args = parser.parse_args()
    result = {
        "python": sys.version,
        "executable": sys.executable,
        "torch_file": torch.__file__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "transformers", "numpy", "pandas", "safetensors")},
        "passed": False,
    }
    if result["cuda_available"]:
        result["gpu"] = torch.cuda.get_device_name(0)
        result["capability"] = torch.cuda.get_device_capability(0)
        result["compiled_architectures"] = torch.cuda.get_arch_list()
        torch.manual_seed(20261014)
        start = time.perf_counter()
        x = torch.randn((256, 256), device="cuda", dtype=torch.float32)
        y = x @ x.T
        torch.cuda.synchronize()
        result["kernel_elapsed_s"] = time.perf_counter() - start
        result["finite_cuda_output"] = bool(torch.isfinite(y).all().item())
        result["passed"] = result["finite_cuda_output"] and "cu128" in torch.__version__ and tuple(result["capability"]) == (12, 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
