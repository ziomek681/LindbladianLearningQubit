import numpy as np
import torch
from torch.utils.data import DataLoader
import time

import dataset as ds
from model import PINN

# Physics setup (must match the data-generation notebook!)
eV = 1.6 * 1e-19
ps = 1e-12

J = 0 * 1e-3 * eV
dt = 0.1 * ps          # must match the dt used to generate the dataset, since its not saved in the dataset metadata

H = np.array([
    [1,  0,  0, 0],
    [0, -1,  2, 0],
    [0,  2, -1, 0],
    [0,  0,  0, 1]
], dtype=complex) * J / 4

I2 = np.eye(2, dtype=complex)
L1 = np.array([[0, 1], [0, 0]], dtype=complex)   # relaxation
L2 = np.array([[1, 0], [0, -1]], dtype=complex)  # dephasing

L_ops = [
    np.kron(L1, I2),
    np.kron(I2, L1),
    np.kron(L2, I2),
    np.kron(I2, L2),
    np.kron(I2, L2) + np.kron(L2, I2),
]


# Worker seeding -- __getitem__ calls np.random.randint, so DataLoader workers need distinct seeds or they'll all draw identical t_idx streams.

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)


def lambda_phys_schedule(epoch, warmup_epochs=10, min_lambda=0.0, max_lambda=1.0):
    """
    Linear warmup of the physics-loss weight: start at 0 so the network first learns a reasonable tau estimate from data alone
    """
    return min_lambda + (max_lambda - min_lambda) * min(1.0, epoch / max(1, warmup_epochs))


def run_epoch(pinn, loader, optimizer, lambda_phys, lambda_data, device, train: bool):
    pinn.train(train)

    total_loss_sum = 0.0
    data_loss_sum = 0.0
    phys_loss_sum = 0.0
    n_batches = 0

    for batch in loader:
        features, labels, rho_t, rho_tp1 = batch
        features = features.to(device)
        labels = labels.to(device)
        rho_t = rho_t.to(device)
        rho_tp1 = rho_tp1.to(device)

        with torch.set_grad_enabled(train):
            loss, loss_data, loss_phys = pinn.total_loss(
                (features, labels, rho_t, rho_tp1), lambda_phys=lambda_phys, lambda_data=lambda_data
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss_sum += loss.item()
        data_loss_sum += loss_data.item()
        phys_loss_sum += loss_phys.item()
        n_batches += 1



    return (
        total_loss_sum / n_batches,
        data_loss_sum / n_batches,
        phys_loss_sum / n_batches,
    )


def train_pinn(
        root,
        n_epochs=100,
        batch_size=64,
        lr=1e-3,
        warmup_epochs=10,
        min_lambda_phys=1.0,
        max_lambda_phys=1.0,
        lambda_data=1.0,
        hidden_dims=(256, 256, 256),
        num_workers=4,
        checkpoint_path="best_pinn.pt",
        seed=0,
        preloaddata=True,
        featuretimes=np.arange(0, 40*1e-12, 2*1e-12),
        device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    featuretimes.sort()

    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = ds.TrajectoryDataset(root, preloaddata, featuretimes)
    train_ds, val_ds, test_ds = ds.split_dataset(dataset, seed=seed)

    print(f"dataset size: {len(dataset)}  "
          f"(train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)})")

    mean, std, labels = ds.compute_label_stats(dataset, train_ds.indices)
    print(f"train label (log-tau) mean: {mean}")
    print(f"train label (log-tau) std:  {std}")
    #print(f"label tau-log:  {labels}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, worker_init_fn=worker_init_fn, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, worker_init_fn=worker_init_fn, persistent_workers=True
    )

    pinn = PINN(
        feature_dim=dataset.feature_dim,
        H=H,
        L_ops=L_ops,
        dt=dt,
        hidden_dims=hidden_dims,
    ).to(device)

    optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)

    best_val_loss = float("inf")

    # history bookkeeping for plotting
    history = {
        "epoch": [],
        "lambda_phys": [],
        "train_total": [], "train_data": [], "train_phys": [],
        "val_total": [], "val_data": [], "val_phys": [],
        "best_epoch": None,
    }

    for epoch in range(n_epochs):
        lambda_phys = lambda_phys_schedule(epoch, warmup_epochs, min_lambda_phys, max_lambda_phys)

        t0 = time.perf_counter()
        train_loss, train_data_loss, train_phys_loss = run_epoch(
            pinn, train_loader, optimizer, lambda_phys, lambda_data, device, train=True
        )
        t1 = time.perf_counter()
        val_loss, val_data_loss, val_phys_loss = run_epoch(
            pinn, val_loader, optimizer, lambda_phys, lambda_data, device, train=False
        )
        t2 = time.perf_counter()

        if epoch % 10 == 0:
            print(
                f"data={t1 - t0:.3f}s, "
                f"validation={t2 - t1:.3f}s, "
            )
            print(
                f"epoch {epoch:03d} | lambda_phys={lambda_phys:.3f} | "
                f"train: total={train_loss:.4e} data={train_data_loss:.4e} phys={train_phys_loss:.4e} | "
                f"val: total={val_loss:.4e} data={val_data_loss:.4e} phys={val_phys_loss:.4e}"
            )

        history["epoch"].append(epoch)
        history["lambda_phys"].append(lambda_phys)
        history["train_total"].append(train_loss)
        history["train_data"].append(train_data_loss)
        history["train_phys"].append(train_phys_loss)
        history["val_total"].append(val_loss)
        history["val_data"].append(val_data_loss)
        history["val_phys"].append(val_phys_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            torch.save(pinn.state_dict(), checkpoint_path)
            print(f"  -> new best val loss, checkpoint saved to {checkpoint_path}")

    pinn.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return pinn, dataset, (train_ds, val_ds, test_ds), history, device