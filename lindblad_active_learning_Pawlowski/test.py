#!/usr/bin/env python3

import argparse
import copy
import json
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from argparse import Namespace

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


from train import parse_args, prepare_bayesian_designs, make_batch, evaluate, test


def main():
    args = parse_args()

    out_path = os.path.join("Results", args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        os.path.join(out_path, "checkpoint.pt"),
        map_location=device,
        weights_only=False,
    )

    if args.use_train_args == True:
        args = checkpoint["args"]

    args = Namespace(**args)

        
    print("args:")
    print(vars(args))

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    adaptive_fraction = 1.0 - args.p_uniform - args.p_bayes
    print("device:", device)

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
    its, iva, itr = perm[:ntest], perm[ntest:nval + ntest], perm[nval + ntest:]
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

    print("Importing model parameters from training", flush=True)
    model.load_state_dict(checkpoint["model_state"])

    theta_train = theta_train.to(device)
    Ftrain = Ftrain.to(device)
    p_train = p_train.to(device)

    theta_val = theta_val.to(device)
    Fval = Fval.to(device)
    p_val = p_val.to(device)

    theta_test = theta_test.to(device)
    Ftest = Ftest.to(device)
    p_test = p_test.to(device)

    best_epoch = checkpoint["epochs"]

    out_path = os.path.join("Results", args.out)
    os.makedirs(out_path, exist_ok=True)

    best_metric = checkpoint["best_val_metric"]
    p_mean = checkpoint["p_mean"].to(device)
    p_std = checkpoint["p_std"].to(device)

    test(best_epoch, theta_test, p_test, Ftest, model, best_metric, adaptive_fraction,
         bayes_counts, bayes_weights, p_mean, p_std, args, device, True)


if __name__ == "__main__":
    main()
