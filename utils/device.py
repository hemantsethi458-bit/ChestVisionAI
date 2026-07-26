"""Device selection utilities."""

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Select the best available compute device.

    Args:
        preference: ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.

    Returns:
        PyTorch device instance.
    """
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    raise ValueError(f"Unsupported device preference: {preference}")
