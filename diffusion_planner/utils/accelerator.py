# Accelerator compatibility layer: CUDA / MUSA(Moore Threads) / CPU.
# Written for the S5000-class (MUSA) training target while keeping the
# upstream CUDA path byte-compatible. All torch.cuda/torch.musa touches in the
# training + planner code go through here.
import torch


def detect_backend(requested: str) -> str:
    """Map a requested device to what is actually usable: cuda | musa | cpu.

    'musa' requires torch_musa; falls back to cpu (with a print) so scripts
    keep running on non-MUSA machines for smoke tests.
    """
    if requested == "musa":
        try:
            import torch_musa  # noqa: F401  (registers torch.musa)
        except ImportError:
            print("[accelerator] torch_musa not installed -> fallback cpu")
            return "cpu"
        if torch.musa.is_available():
            return "musa"
        print("[accelerator] torch.musa not available -> fallback cpu")
        return "cpu"
    if requested == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


def device_count(backend: str) -> int:
    if backend == "cuda":
        return torch.cuda.device_count()
    if backend == "musa":
        return torch.musa.device_count()
    return 0


def set_device(backend: str, index: int) -> None:
    if backend == "cuda":
        torch.cuda.set_device(index)
    elif backend == "musa":
        torch.musa.set_device(index)


def synchronize(backend: str) -> None:
    if backend == "cuda":
        torch.cuda.synchronize()
    elif backend == "musa":
        torch.musa.synchronize()


def dist_backend(accel: str) -> str:
    """torch.distributed backend name per accelerator."""
    if accel == "cuda":
        return "nccl"
    if accel == "musa":
        return "mccl"
    return "gloo"


def rank_device(accel: str, rank: int):
    """torch.device for this rank: cuda:N / musa:N / cpu.

    Using torch.device('musa:N') instead of a bare int is the portable form
    for torch_musa (a bare int resolves against the cuda backend).
    """
    if accel == "cpu":
        return torch.device("cpu")
    return torch.device(f"{accel}:{rank}")


def assert_available(accel: str) -> None:
    if accel == "cuda":
        assert torch.cuda.is_available(), "cuda is not available"
    elif accel == "musa":
        assert torch.musa.is_available(), "musa is not available (torch_musa installed? GPU visible?)"
