"""Generate the probability and per-setting Fisher caches used by train.py.

The Fisher tensor is computed with the same convention as the original
closed-loop code: finite differences with respect to the normalized parameters
z=(theta-MID)/HALF.
"""
import argparse
from pathlib import Path
import torch

from config import LOW, HIGH, HALF, DEFAULT_TIMES
from physics import build_system, probabilities


@torch.no_grad()
def fisher_settings(theta, times, system, max_step=0.1, eps_z=2e-3):
    """One-shot Bernoulli Fisher per setting, shape [B,S,5,5], wrt z."""
    device = theta.device
    half = HALF.to(device)
    p0 = probabilities(theta, times, system, max_step)
    B, S = p0.shape
    J = torch.empty(B, S, 5, device=device, dtype=torch.float32)

    for k in range(5):
        dtheta = eps_z * half[k]
        tp = theta.clone()
        tm = theta.clone()
        tp[:, k] += dtheta
        tm[:, k] -= dtheta
        pp = probabilities(tp, times, system, max_step)
        pm = probabilities(tm, times, system, max_step)
        # z changes by +/- eps_z, hence denominator 2*eps_z.
        J[:, :, k] = (pp - pm) / (2.0 * eps_z)

    inv_var = 1.0 / (p0 * (1.0 - p0)).clamp_min(1e-8)
    return inv_var[:, :, None, None] * J[:, :, :, None] * J[:, :, None, :]


def generate_caches(probability_path="lindblad_probability_pool.pt",
                    fisher_path="oracle_fisher_settings_pool.pt",
                    n_pool=20000, batch_size=256, seed=1234,
                    times=DEFAULT_TIMES, max_step=0.1, J_meV=0.0,
                    eps_z=2e-3, device=None, overwrite=False):
    probability_path = Path(probability_path)
    fisher_path = Path(fisher_path)

    if not overwrite and probability_path.exists() and fisher_path.exists():
        print("Caches already exist; nothing to do.")
        return

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print("device:", device)
    print("generating cache pool:", n_pool)
    print("times:", list(times))
    print("finite-difference eps_z:", eps_z)

    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n_pool, 5, generator=g)
    theta_all = LOW[None] + (HIGH - LOW)[None] * u

    system = build_system(device, J_meV)
    p_parts, F_parts = [], []

    for lo in range(0, n_pool, batch_size):
        hi = min(lo + batch_size, n_pool)
        theta = theta_all[lo:hi].to(device)
        p = probabilities(theta, times, system, max_step)
        Fs = fisher_settings(theta, times, system, max_step, eps_z)
        p_parts.append(p.cpu())
        F_parts.append(Fs.cpu())
        print(f"  {hi:6d}/{n_pool}", flush=True)

    p_all = torch.cat(p_parts)
    F_all = torch.cat(F_parts)

    probability_path.parent.mkdir(parents=True, exist_ok=True)
    fisher_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "theta": theta_all.float(),
        "p": p_all.float(),
        "times": list(times),
        "J_meV": float(J_meV),
        "max_step": float(max_step),
        "seed": int(seed),
    }, probability_path)

    torch.save({
        "theta": theta_all.float(),
        "Fset": F_all.float(),
        "times": list(times),
        "eps_z": float(eps_z),
        "derivative_coordinates": "z=(theta-MID)/HALF",
        "J_meV": float(J_meV),
        "max_step": float(max_step),
        "seed": int(seed),
    }, fisher_path)

    print("saved:", probability_path)
    print("saved:", fisher_path)
    print("p shape:", tuple(p_all.shape))
    print("Fset shape:", tuple(F_all.shape))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="lindblad_probability_pool.pt")
    p.add_argument("--fisher-cache", default="oracle_fisher_settings_pool.pt")
    p.add_argument("--n-pool", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--times", type=float, nargs="+", default=list(DEFAULT_TIMES))
    p.add_argument("--max-step", type=float, default=0.10)
    p.add_argument("--J-meV", type=float, default=0.0)
    p.add_argument("--eps-z", type=float, default=2e-3)
    p.add_argument("--device", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    generate_caches(
        a.cache, a.fisher_cache, a.n_pool, a.batch_size, a.seed,
        a.times, a.max_step, a.J_meV, a.eps_z, a.device, a.overwrite,
    )


if __name__ == "__main__":
    main()
