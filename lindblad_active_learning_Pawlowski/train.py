#!/usr/bin/env python3

"""
Current clean training pipeline.

Training distribution:
    uniform + static Bayesian + closed-loop adaptive

Deployment:
    global Bayesian warmup
    -> Transformer theta_hat
    -> accumulated local Fisher
    -> more shots
    -> repeat

Loss:
    supervised normalized-parameter MSE
    + lambda_phys * differentiable binomial Lindblad NLL

Oracle is evaluation-only.
"""
import argparse
import copy
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import DEFAULT_TIMES, DEFAULT_KS, DEFAULT_ADAPTIVE_KS
from decoder import LindbladDecoder
from model import theta_to_z, z_to_theta
from measurements import features_from_counts, weights_to_counts
from design import (
    bayesian_aopt_weights, static_batch, sequential_batch
)
from physics import build_system, binomial_physics_nll
from io_utils import (
    load_probability_pool, load_fisher_pool,
    rmse_score, true_fisher_ascore,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="lindblad_probability_pool.pt")
    p.add_argument("--fisher-cache", default="oracle_fisher_settings_pool.pt")
    p.add_argument("--checkpoint", default="adaptive_phys_transformer_clean.pt")
    p.add_argument("--n-pool", type=int, default=20000,
                   help="Pool size used only when caches must be generated.")
    p.add_argument("--cache-batch", type=int, default=256,
                   help="Batch size used only for cache generation.")
    p.add_argument("--eps-z", type=float, default=2e-3,
                   help="Finite-difference step in normalized z for Fisher cache.")
    p.add_argument("--regenerate-cache", action="store_true",
                   help="Force regeneration of probability and Fisher caches.")
    p.add_argument("--out", default="adaptive_phys_transformer_clean_results.json")

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--train-Ks", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument("--Ks", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument("--adaptive-Ks", type=int, nargs="+",
                   default=list(DEFAULT_ADAPTIVE_KS))
    p.add_argument("--warmup", type=int, default=4096)

    # Current successful training emphasis: 10% U, 20% B, 70% A.
    p.add_argument("--p-uniform", type=float, default=0.10)
    p.add_argument("--p-bayes", type=float, default=0.00)

    p.add_argument("--epsilon", type=float, default=0.20)
    p.add_argument("--bayes-samples", type=int, default=2048)
    p.add_argument("--design-iters", type=int, default=32)
    p.add_argument("--prior-precision", type=float, default=3.0)
    p.add_argument("--explore", type=float, default=0.10)
    p.add_argument("--bayes-chunk", type=int, default=512)

    p.add_argument("--lambda-phys", type=float, default=0.20)
    p.add_argument("--phys-every", type=int, default=4)
    p.add_argument("--phys-batch", type=int, default=32)
    p.add_argument("--times", type=float, nargs="+", default=list(DEFAULT_TIMES))
    p.add_argument("--max-step", type=float, default=0.10)
    p.add_argument("--J-meV", type=float, default=0.0)

    p.add_argument("--n-test", type=int, default=512)
    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()


def prepare_bayesian_designs(Ftrain, all_Ks, args, device):
    generator = torch.Generator().manual_seed(args.seed + 991)
    M = min(args.bayes_samples, len(Ftrain))
    idx = torch.randperm(len(Ftrain), generator=generator)[:M]
    Fmc = Ftrain[idx].to(device)

    weights, counts = {}, {}
    for K in all_Ks:
        print(f"Bayesian allocation K={K}")
        w = bayesian_aopt_weights(
            Fmc, K, args.prior_precision, args.design_iters,
            args.explore, args.bayes_chunk,
        ).cpu()
        weights[K] = w
        counts[K] = weights_to_counts(w, K)[0].cpu()
    return weights, counts


def make_batch(mode, model, p, K, theta_train, Ftrain,
               bayes_counts, bayes_weights, p_mean, p_std, args, device,
               Ftrue=None):
    if mode in ("uniform", "bayes"):
        return static_batch(p, K, mode, bayes_counts, device)

    return sequential_batch(
        model, p, K, theta_train, Ftrain,
        bayes_counts, bayes_weights, p_mean, p_std,
        args.warmup, args.adaptive_Ks, args.epsilon,
        args.prior_precision, args.design_iters, device,
        Ftrue=Ftrue if mode == "oracle" else None,
    )


@torch.no_grad()
def evaluate(model, theta, p, Ftrue, K, mode, theta_train, Ftrain,
             bayes_counts, bayes_weights, p_mean, p_std, args, device):
    model.eval()
    predictions, allocations = [], []

    for lo in range(0, len(theta), args.eval_batch):
        pp = p[lo:lo + args.eval_batch].to(device)
        Ft = Ftrue[lo:lo + args.eval_batch]

        n, N = make_batch(
            mode, model, pp, K, theta_train, Ftrain,
            bayes_counts, bayes_weights, p_mean, p_std, args, device,
            Ftrue=Ft,
        )
        x = features_from_counts(n, N, K, p_mean, p_std)
        predictions.append(z_to_theta(model(x)).cpu())
        allocations.append(N.cpu())

    pred = torch.cat(predictions)
    N = torch.cat(allocations)
    return rmse_score(theta, pred), N


def main():
    args = parse_args()
    if args.p_uniform + args.p_bayes > 1:
        raise ValueError("p-uniform + p-bayes must be <= 1")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adaptive_fraction = 1.0 - args.p_uniform - args.p_bayes
    print("device:", device)
    print("training mix U/B/A:",
          args.p_uniform, args.p_bayes, adaptive_fraction)

    # A fresh checkout is self-contained: create both caches automatically.
    # If only one cache is present we regenerate both, because theta and Fset
    # must refer to exactly the same parameter pool and ordering.
    cache_missing = not Path(args.cache).exists() or not Path(args.fisher_cache).exists()
    if args.regenerate_cache or cache_missing:
        from generate_cache import generate_caches
        if cache_missing:
            print("cache files not found -> generating them now")
        else:
            print("--regenerate-cache -> rebuilding cache files")
        generate_caches(
            probability_path=args.cache,
            fisher_path=args.fisher_cache,
            n_pool=args.n_pool,
            batch_size=args.cache_batch,
            seed=args.seed,
            times=args.times,
            max_step=args.max_step,
            J_meV=args.J_meV,
            eps_z=args.eps_z,
            device=device,
            overwrite=True,
        )

    print("Loading data", flush=True)
    theta, p = load_probability_pool(args.cache)
    Fall = load_fisher_pool(args.fisher_cache)
    if len(Fall) != len(theta):
        raise RuntimeError("Fisher/probability cache length mismatch.")
    if p.shape[1] != 16 * len(args.times):
        raise RuntimeError("Cache/settings mismatch with --times.")

    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(theta), generator=generator)
    nval = max(1, int(0.2 * len(theta)))
    iv, it = perm[:nval], perm[nval:]
    theta_train, p_train, Ftrain = theta[it], p[it], Fall[it]
    theta_val, p_val, Fval = theta[iv], p[iv], Fall[iv]

    p_mean = p_train.mean(dim=0)
    p_std = p_train.std(dim=0).clamp_min(1e-8)
    n_settings = p.shape[1]

    all_Ks = sorted(set(
        args.train_Ks + args.Ks + args.adaptive_Ks + [args.warmup]
    ))
    bayes_weights, bayes_counts = prepare_bayesian_designs(
        Ftrain, all_Ks, args, device
    )

    loader = DataLoader(
        TensorDataset(theta_train, p_train),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )

    print("Building model", flush=True)
    model = LindbladDecoder(n_settings).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-6
    )
    system = build_system(device, args.J_meV)

    best_metric = float("inf")
    best_state = None

    theta_train = theta_train.to(device)
    Ftrain = Ftrain.to(device)
    print("Starting training", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        iterator = iter(loader)
        total_data = total_phys = 0.0
        n_phys = 0
        modes = {"U": 0, "B": 0, "A": 0}

        for step in range(args.steps_per_epoch):
            try:
                th, pp = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                th, pp = next(iterator)

            th, pp = th.to(device), pp.to(device)
            K = args.train_Ks[
                torch.randint(len(args.train_Ks), (1,)).item()
            ]

            r = torch.rand(()).item()
            if r < args.p_uniform:
                mode = "U"
                n, N = static_batch(pp, K, "uniform", bayes_counts, device)
            elif r < args.p_uniform + args.p_bayes:
                mode = "B"
                n, N = static_batch(pp, K, "bayes", bayes_counts, device)
            else:
                mode = "A"
                model.eval()
                n, N = make_batch(
                    "adaptive", model, pp, K, theta_train, Ftrain,
                    bayes_counts, bayes_weights, p_mean, p_std, args, device
                )
                model.train()
            modes[mode] += 1

            x = features_from_counts(n, N, K, p_mean, p_std)
            z_pred = model(x)
            loss_data = F.mse_loss(z_pred, theta_to_z(th))
            loss = loss_data

            if args.lambda_phys > 0 and step % args.phys_every == 0:
                q = min(args.phys_batch, len(th))
                idx = torch.randperm(len(th), device=device)[:q]
                theta_pred = z_to_theta(z_pred[idx])
                loss_phys = binomial_physics_nll(
                    theta_pred, n[idx], N[idx], system,
                    args.times, args.max_step,
                )
                loss = loss + args.lambda_phys * loss_phys
                total_phys += loss_phys.item()
                n_phys += 1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_data += loss_data.item()

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            ncheck = min(256, len(theta_val))
            Kcheck = 8192
            scores = {}
            for mode in ("uniform", "bayes", "adaptive"):
                (s, _), _ = evaluate(
                    model, theta_val[:ncheck], p_val[:ncheck], Fval[:ncheck],
                    Kcheck, mode, theta_train, Ftrain,
                    bayes_counts, bayes_weights, p_mean, p_std, args, device,
                )
                scores[mode] = s

            metric = sum(scores.values()) / 3
            print(
                f"epoch {epoch:4d} "
                f"data={total_data/args.steps_per_epoch:.5e} "
                f"phys={total_phys/max(n_phys,1):.5e} modes={modes} | "
                f"val@{Kcheck}: U={scores['uniform']:.4f} "
                f"B={scores['bayes']:.4f} A={scores['adaptive']:.4f}", flush=True
            )
            if metric < best_metric:
                best_metric = metric
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        "model_state": model.state_dict(),
        "p_mean": p_mean.cpu(),
        "p_std": p_std.cpu(),
        "bayes_counts": bayes_counts,
        "args": vars(args),
        "best_val_metric": best_metric,
    }, args.checkpoint)
    print("Saved checkpoint:", args.checkpoint)

    ntest = min(args.n_test, len(theta_val))
    th, pp, Ft = theta_val[:ntest], p_val[:ntest], Fval[:ntest]
    rows, fisher_rows = [], []

    print("\nFINAL DECODER RMSE (lower = better)")
    print(f"{'K':>8} {'uniform':>12} {'bayes':>12} {'adaptive':>12} {'oracle':>12}")

    for K in args.Ks:
        result = {}
        allocation = {}
        for mode in ("uniform", "bayes", "adaptive", "oracle"):
            result[mode], allocation[mode] = evaluate(
                model, th, pp, Ft, K, mode, theta_train, Ftrain,
                bayes_counts, bayes_weights, p_mean, p_std, args, device,
            )

        print(
            f"{K:8d} "
            f"{result['uniform'][0]:12.6f} "
            f"{result['bayes'][0]:12.6f} "
            f"{result['adaptive'][0]:12.6f} "
            f"{result['oracle'][0]:12.6f}"
        )

        rows.append({
            "K": K,
            **{m: result[m][0] for m in result},
            **{f"{m}_rmse": [float(x) for x in result[m][1]]
               for m in result},
        })

        fisher = {
            mode: true_fisher_ascore(
                Ft, allocation[mode], args.prior_precision
            )
            for mode in allocation
        }
        fisher_rows.append({
            "K": K,
            **{f"{m}_mean": fisher[m][0] for m in fisher},
            **{f"{m}_median": fisher[m][1] for m in fisher},
        })

    print("\nTRUE-FISHER A-SCORE (lower = better)")
    print(f"{'K':>8} {'uniform':>14} {'bayes':>14} {'adaptive':>14} {'oracle':>14}")
    for r in fisher_rows:
        print(
            f"{r['K']:8d} {r['uniform_mean']:14.6e} "
            f"{r['bayes_mean']:14.6e} {r['adaptive_mean']:14.6e} "
            f"{r['oracle_mean']:14.6e}"
        )

    with open(args.out, "w") as f:
        json.dump({
            "rows": rows,
            "true_fisher_rows": fisher_rows,
            "best_val_metric": best_metric,
            "training_mix": {
                "uniform": args.p_uniform,
                "bayes": args.p_bayes,
                "adaptive": adaptive_fraction,
            },
            "lambda_phys": args.lambda_phys,
            "epsilon": args.epsilon,
            "warmup": args.warmup,
        }, f, indent=2)
    print("Saved results:", args.out)


if __name__ == "__main__":
    main()
