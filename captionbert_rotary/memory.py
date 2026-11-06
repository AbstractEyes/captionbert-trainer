"""Conservative CUDA allocator cap accounting for other device users."""
import subprocess
import torch


def device_used_bytes(device=0):
    free, total = torch.cuda.mem_get_info(device)
    measured = total - free
    # WDDM reports application use differently; take the larger observation.
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(device)],
            text=True, timeout=5, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        measured = max(measured, int(out.strip()) * 1024 ** 2)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return measured


class MemoryGuard:
    def __init__(self, limit_gb=20.0, device="cuda"):
        self.enabled = str(device).startswith("cuda")
        self.limit = int(limit_gb * 1e9)
        self.device = torch.device(device).index or 0
        self.max_device_used = 0
        if not self.enabled:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu for unit diagnostics")
        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()
        _, total = torch.cuda.mem_get_info(self.device)
        # The 4090 always keeps the user's ceiling even if a Colab config is reused.
        if "4090" in torch.cuda.get_device_name(self.device):
            self.limit = min(self.limit, 20_000_000_000)
        other = max(0, device_used_bytes(self.device) - torch.cuda.memory_reserved(self.device))
        allowance = min(total, self.limit) - other - 768 * 1024 ** 2
        if allowance < 1024 ** 3:
            raise RuntimeError("Insufficient GPU headroom under the total-device memory ceiling")
        torch.cuda.set_per_process_memory_fraction(allowance / total, self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        print(f"GPU memory ceiling {self.limit / 1e9:.2f} GB total; "
              f"allocator allowance {allowance / 1e9:.2f} GB", flush=True)
        self.check()

    def check(self):
        if not self.enabled:
            return
        used = device_used_bytes(self.device)
        self.max_device_used = max(self.max_device_used, used)
        if used >= self.limit:
            raise RuntimeError("Total device memory reached the configured ceiling; stop other GPU jobs")

    def report(self):
        if not self.enabled:
            return {}
        self.check()
        return {"peak_allocated_gb": torch.cuda.max_memory_allocated(self.device) / 1e9,
                "peak_reserved_gb": torch.cuda.max_memory_reserved(self.device) / 1e9,
                "max_observed_device_gb": self.max_device_used / 1e9,
                "total_device_limit_gb": self.limit / 1e9}
