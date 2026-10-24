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
import os
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
from plotting import PlotCorelations, PlotErrorColor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="lindblad_probability_pool.pt")
    p.add_argument("--fisher-cache", default="oracle_fisher_settings_pool.pt")
    p.add_argument("--n-pool", type=int, default=20000,
                   help="Pool size used only when caches must be generated.")
    p.add_argument("--cache-batch", type=int, default=256,
                   help="Batch size used only for cache generation.")
    p.add_argument("--eps-z", type=float, default=2e-3,
                   help="Finite-difference step in normalized z for Fisher cache.")
    p.add_argument("--regenerate-cache", action="store_true",
                   help="Force regeneration of probability and Fisher caches.")
    p.add_argument("--out", default="Run0")

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--steps-per-epoch", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--eval-batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-metric", default="all")

    p.add_argument("--test-Ks", type=int, nargs="+", default=131072)
    p.add_argument("--val-Ks", type=int, nargs="+", default=16384)
    p.add_argument("--train-Ks", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument("--Ks", type=int, nargs="+", default=list(DEFAULT_KS))
    p.add_argument("--adaptive-Ks", type=int, nargs="+",
                   default=list(DEFAULT_ADAPTIVE_KS))
    p.add_argument("--warmup", type=int, default=4096)
    p.add_argument("--warmup-uniform", action="store_true")

    # Current successful training emphasis: 10% U, 20% B, 70% A.
    p.add_argument("--p-uniform", type=float, default=0.00)
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
        args.warmup, args.warmup_uniform,
        args.adaptive_Ks, args.epsilon,
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
        predictions.append(z_to_theta(model(x)))
        allocations.append(N.cpu())

    pred = torch.cat(predictions)
    N = torch.cat(allocations)
    return rmse_score(theta.cpu(), pred.cpu()), N

@torch.no_grad()
def test(epoch, theta_test, p_test, Ftest, model, best_metric, adaptive_fraction,
	 bayes_counts, bayes_weights, p_mean, p_std, args, device, FinalTest = True):
    ntest = min(args.n_test, len(theta_test))
    th, pp, Ft = theta_test[:ntest], p_test[:ntest], Ftest[:ntest]
    rows, fisher_rows = [], []

    if FinalTest == True:
        print("\nFINAL DECODER RMSE (lower = better)")
        print(f"{'K':>8} {'uniform':>12} {'bayes':>12} {'adaptive':>12} {'oracle':>12}")

    for K in args.Ks:
        result = {}
        allocation = {}
        for mode in ("uniform", "bayes", "adaptive", "oracle"):
            result[mode], allocation[mode] = evaluate(
                model, th, pp, Ft, K, mode, theta_test, Ftest,
                bayes_counts, bayes_weights, p_mean, p_std, args, device,
            )
        if FinalTest == True:
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
                Ft.cpu(), allocation[mode], args.prior_precision
            )
            for mode in allocation
        }
        fisher_rows.append({
            "K": K,
            **{f"{m}_mean": fisher[m][0] for m in fisher},
            **{f"{m}_median": fisher[m][1] for m in fisher},
        })

    if FinalTest == True:
        print("\nTRUE-FISHER A-SCORE (lower = better)")
        print(f"{'K':>8} {'uniform':>14} {'bayes':>14} {'adaptive':>14} {'oracle':>14}")
        for r in fisher_rows:
            print(
                f"{r['K']:8d} {r['uniform_mean']:14.6e} "
                f"{r['bayes_mean']:14.6e} {r['adaptive_mean']:14.6e} "
                f"{r['oracle_mean']:14.6e}"
            )

    out_path = os.path.join("Results", args.out)
    os.makedirs(out_path, exist_ok=True)

    with open(os.path.join(out_path, "data.json"), "w") as f:
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
            "epochs": epoch,
        }, f, indent=2)

    if FinalTest == True:
        n, N = make_batch(
            "adaptive", model, pp, args.test_Ks, theta_test, Ftest,
            bayes_counts, bayes_weights, p_mean, p_std, args, device,
            Ftrue=Ft,
        )
        x = features_from_counts(n, N, args.test_Ks, p_mean, p_std)
        theta_pred = z_to_theta(model(x)).cpu()
        PlotCorelations(0, out_path, th.detach().cpu().numpy(), theta_pred.detach().cpu().numpy())
        PlotCorelations(1, out_path, th.detach().cpu().numpy(), theta_pred.detach().cpu().numpy())
        PlotCorelations(2, out_path, th.detach().cpu().numpy(), theta_pred.detach().cpu().numpy())
        PlotCorelations(3, out_path, th.detach().cpu().numpy(), theta_pred.detach().cpu().numpy())
        PlotCorelations(4, out_path, th.detach().cpu().numpy(), theta_pred.detach().cpu().numpy())

    if FinalTest == True:
        print("Saved results:", args.out)


def main():
    args = parse_args()
    print("args:")
    print(vars(args))
    if args.p_uniform + args.p_bayes > 1:
        raise ValueError("p-uniform + p-bayes must be <= 1")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
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
    ntest = max(1, int(0.1 * len(theta)))
    its, iva, itr = perm[:ntest], perm[ntest:nval+ntest], perm[nval+ntest:]
    theta_train, p_train, Ftrain = theta[itr], p[itr], Fall[itr]
    theta_val, p_val, Fval = theta[iva], p[iva], Fall[iva]
    theta_test, p_test, Ftest = theta[its], p[its], Fall[its]

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
    p_train = p_train.to(device)

    theta_val = theta_val.to(device)
    Fval = Fval.to(device)
    p_val = p_val.to(device)

    theta_test = theta_test.to(device)
    Ftest = Ftest.to(device)
    p_test = p_test.to(device)

    best_epoch = 0
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
            #ncheck = min(256, len(theta_val))
            Kcheck = args.val_Ks
            scores = {}
            for mode in ("uniform", "bayes", "adaptive"):
                (s, _), _ = evaluate(
                    model, theta_val, p_val, Fval,
                    Kcheck, mode, theta_train, Ftrain,
                    bayes_counts, bayes_weights, p_mean, p_std, args, device,
                )
                scores[mode] = s

            if args.val_metric == "all":
            	metric = sum(scores.values()) / 3
            else:
            	metric = scores[args.val_metric]
            print(
                f"epoch {epoch:4d} "
                f"data={total_data/args.steps_per_epoch:.5e} "
                f"phys={total_phys/max(n_phys,1):.5e} modes={modes} | "
                f"val@{Kcheck}: U={scores['uniform']:.4f} "
                f"B={scores['bayes']:.4f} A={scores['adaptive']:.4f}", flush=True
            )
            if metric < best_metric:
                print("Saved model", flush=True)
                best_epoch = epoch
                best_metric = metric
                best_state = copy.deepcopy(model.state_dict())

                out_path = os.path.join("Results", args.out)
                os.makedirs(out_path, exist_ok=True)
                torch.save({
                    "model_state": model.state_dict(),
                    "p_mean": p_mean.cpu(),
                    "p_std": p_std.cpu(),
                    "bayes_counts": bayes_counts,
                    "args": vars(args),
                    "best_val_metric": best_metric,
                    "epochs": best_epoch,
                }, os.path.join(out_path, "checkpoint.pt"))
                test(best_epoch, theta_val, p_val, Fval, model, best_metric, adaptive_fraction,
	 		bayes_counts, bayes_weights, p_mean, p_std, args, device, False)


    if best_state is not None:
        model.load_state_dict(best_state)

    out_path = os.path.join("Results", args.out)
    os.makedirs(out_path, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "p_mean": p_mean.cpu(),
        "p_std": p_std.cpu(),
        "bayes_counts": bayes_counts,
        "args": vars(args),
        "best_val_metric": best_metric,
        "epochs": best_epoch,
    }, os.path.join(out_path, "checkpoint.pt"))
    test(best_epoch, theta_test, p_test, Ftest, model, best_metric, adaptive_fraction,
	 		bayes_counts, bayes_weights, p_mean, p_std, args, device, True)



if __name__ == "__main__":
    main()
