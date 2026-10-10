"""Cache loading and metrics."""
import math
import torch
from config import LOW, HIGH


def load_probability_pool(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        theta = next((obj[k] for k in ("theta", "theta_pool", "thetas") if k in obj), None)
        p = next((obj[k] for k in ("p", "p_pool", "probs", "probabilities") if k in obj), None)
    else:
        theta, p = obj[:2]
    if theta is None or p is None:
        raise KeyError("Cannot identify theta/probability tensors.")
    return torch.as_tensor(theta).float(), torch.as_tensor(p).float()


def load_fisher_pool(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    Fset = obj["Fset"] if isinstance(obj, dict) else obj
    return torch.as_tensor(Fset).float()


def rmse_score(theta_true, theta_pred):
    rmse = torch.sqrt(((theta_pred - theta_true) ** 2).mean(dim=0))
    prior_std = (HIGH - LOW) / math.sqrt(12.0)
    normalized = torch.sqrt(((rmse / prior_std) ** 2).mean()).item()
    return normalized, rmse


@torch.no_grad()
def true_fisher_ascore(Ftrue, N, prior_precision):
    I = torch.eye(5, dtype=Ftrue.dtype)[None]
    Ftot = prior_precision * I + (
        N.float()[:, :, None, None] * Ftrue.float()
    ).sum(dim=1)
    Finv = torch.linalg.pinv(Ftot, rcond=1e-6)
    A = torch.diagonal(Finv, dim1=-2, dim2=-1).sum(dim=-1)
    return A.mean().item(), A.median().item()
