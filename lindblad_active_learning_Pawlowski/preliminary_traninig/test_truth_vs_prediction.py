#!/usr/bin/env python3
"""
test_trajectories.py

Small post-training diagnostic:
for a few held-out examples, follow ONE closed-loop adaptive experiment
4096 -> 8192 -> 16384 -> 32768 -> 65536 and compare theta_hat(K)
with theta_true.

Run from the directory containing:
  train_transformer_adaptive_phys_v3_oracle.py
  adaptive_phys_transformer_v3.pt
  lindblad_probability_pool.pt
  oracle_fisher_settings_pool.pt

Example:
  python test_trajectories.py --n-examples 12
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

import lindblad_active_learning.preliminary_traninig.train_transformer_adaptive as core


PARAM_NAMES = ["T1 q1", "T1 q2", "Tphi q1", "Tphi q2", "tau corr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="lindblad_probability_pool.pt")
    ap.add_argument("--fisher-cache", default="oracle_fisher_settings_pool.pt")
    ap.add_argument("--checkpoint", default="adaptive_phys_transformer_v3.pt")
    ap.add_argument("--n-examples", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="adaptive_prediction_trajectories")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # ----- data and exactly the same train/validation split as training -----
    theta, p = core.load_pool(args.cache)
    fo = torch.load(args.fisher_cache, map_location="cpu", weights_only=False)
    Fall = torch.as_tensor(fo["Fset"] if isinstance(fo, dict) else fo).float()

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(theta), generator=g)
    nv = max(1, int(0.2 * len(theta)))
    iv, it = perm[:nv], perm[nv:]
    th_tr, F_tr = theta[it], Fall[it]
    th_te, p_te = theta[iv], p[iv]

    # ----- trained decoder -----
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    S = int(ck["S"])
    model = core.MeasurementTransformer(S).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    p_mean = ck["p_mean"].float()
    p_std = ck["p_std"].float()

    saved_args = ck.get("args", {})
    warmup = int(saved_args.get("warmup", 4096))
    epsilon = float(saved_args.get("epsilon", 0.20))
    prior_precision = float(saved_args.get("prior_precision", 3.0))
    design_iters = int(saved_args.get("design_iters", 32))
    Ks = [4096, 8192, 16384, 32768, 65536]
    adaptive_Ks = [k for k in Ks if k > warmup]

    # Bayesian counts were saved in the checkpoint.
    # Their normalized integer allocations are sufficient here as w_B.
    bc_raw = ck["bayes_counts"]
    bayes_counts = {}
    for k, v in bc_raw.items():
        bayes_counts[int(k)] = torch.as_tensor(v).long()
    bayes_weights = {
        K: bayes_counts[K].float() / float(bayes_counts[K].sum())
        for K in Ks
    }

    n_examples = min(args.n_examples, len(th_te))
    truths = th_te[:n_examples].clone()
    trajectories = torch.empty(n_examples, len(Ks), 5)

    # ----- one genuine sequential trajectory per test example -----
    with torch.no_grad():
        for i in range(n_examples):
            pp = p_te[i:i+1].to(device)

            Ntot = bayes_counts[warmup].to(device)[None].clone()
            ntot = core.sample_counts(pp, Ntot)

            def predict(K):
                x = core.features_from_counts(ntot, Ntot, K, p_mean, p_std)
                return core.z_to_theta(model(x))

            pred = predict(warmup)
            trajectories[i, 0] = pred[0].cpu()

            prev = warmup
            for j, K in enumerate(adaptive_Ks, start=1):
                delta = K - prev

                Flocal = core.nearest_fisher(
                    pred.cpu(), th_tr, F_tr
                ).to(device)

                wlocal = core.sequential_local_aopt_weights(
                    Flocal, Ntot, delta, prior_precision, design_iters
                )

                wb = bayes_weights[K].to(device)[None]
                wnew = (1.0 - epsilon) * wlocal + epsilon * wb
                wnew = wnew / wnew.sum(1, keepdim=True)

                Nnew = core.weights_to_counts(wnew, delta).to(device)
                ntot += core.sample_counts(pp, Nnew)
                Ntot += Nnew

                pred = predict(K)
                trajectories[i, j] = pred[0].cpu()
                prev = K

    # ----- print ensemble convergence -----
    err = trajectories - truths[:, None, :]
    rmse = torch.sqrt((err ** 2).mean(0))  # [K, parameter]

    print("\nRMSE of the SAME adaptive trajectories")
    print(f"{'K':>8} " + " ".join(f"{x:>11}" for x in PARAM_NAMES))
    for j, K in enumerate(Ks):
        print(f"{K:8d} " + " ".join(f"{x:11.4f}" for x in rmse[j]))

    print("\nExample 0: true theta")
    print(" ".join(f"{x:.4f}" for x in truths[0]))
    for j, K in enumerate(Ks):
        print(f"K={K:5d}: " + " ".join(f"{x:.4f}" for x in trajectories[0, j]))

    # ----- ground truth vs prediction: one panel per parameter -----
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4))

    # One color per K, shared by all panels.
    colors = plt.cm.viridis(torch.linspace(0.05, 0.95, len(Ks)).numpy())

    for q, ax in enumerate(axes):
        true_q = truths[:, q].numpy()

        # Common limits for truth and all predictions -> meaningful y=x line.
        pred_q = trajectories[:, :, q].numpy()
        lo = min(true_q.min(), pred_q.min())
        hi = max(true_q.max(), pred_q.max())
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        lo, hi = lo - pad, hi + pad

        ax.plot([lo, hi], [lo, hi], "--", linewidth=1.2, label="ideal")

        for j, K in enumerate(Ks):
            ax.scatter(
                true_q,
                pred_q[:, j],
                s=28,
                alpha=0.75,
                color=colors[j],
                label=f"{K//1024}k",
            )

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(PARAM_NAMES[q])
        ax.set_xlabel(r"$\theta_{\rm true}$ [ps]")
        if q == 0:
            ax.set_ylabel(r"$\hat{\theta}$ [ps]")
        ax.grid(alpha=0.2)

        # Show improvement from warm-up to final budget.
        ax.text(
            0.04, 0.96,
            f"RMSE: {rmse[0,q]:.2f} -> {rmse[-1,q]:.2f} ps",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )

    handles, labels = axes[-1].get_legend_handles_labels()
    # Put a single common legend below the panels.
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(Ks) + 1,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )
    fig.suptitle("Ground truth vs adaptive Transformer prediction", y=1.02)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(args.out + "_truth_vs_prediction.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.out + "_truth_vs_prediction.pdf", bbox_inches="tight")
    plt.close(fig)

    # ----- compact machine-readable output -----
    payload = {
        "K": Ks,
        "parameter_names": PARAM_NAMES,
        "theta_true": truths.tolist(),
        "theta_hat": trajectories.tolist(),
        "rmse_by_K": rmse.tolist(),
    }
    Path(args.out + ".json").write_text(json.dumps(payload, indent=2))

    print("\nSaved:")
    print(" ", args.out + "_truth_vs_prediction.png")
    print(" ", args.out + "_truth_vs_prediction.pdf")
    print(" ", args.out + ".json")


if __name__ == "__main__":
    main()
