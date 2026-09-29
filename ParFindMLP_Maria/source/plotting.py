import matplotlib.pyplot as plt
import dataset as ds

import numpy as np
import torch
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from pathlib import Path


def plot_history(history, save_path="pinn_training.png"):
    epochs = history["epoch"]
    best_epoch = history["best_epoch"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # --- 1. total loss ---------------------------------------------------
    ax = axes[0]
    ax.plot(epochs, history["train_total"], label="train", color="#3B82F6")
    ax.plot(epochs, history["val_total"], label="val", color="#F97316")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
                   label=f"best ckpt (epoch {best_epoch})")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("total loss (log scale)")
    ax.set_title("Total loss")
    ax.legend(fontsize=8)

    # --- 2. data vs physics loss ------------------------------------------
    ax = axes[1]
    ax.plot(epochs, history["train_data"], label="train data", color="#3B82F6")
    ax.plot(epochs, history["val_data"], label="val data", color="#F97316")
    ax.plot(epochs, history["train_phys"], label="train phys", color="#3B82F6",
            linestyle="--")
    ax.plot(epochs, history["val_phys"], label="val phys", color="#F97316",
            linestyle="--")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log scale)")
    ax.set_title("Data loss (solid) vs physics loss (dashed)")
    ax.legend(fontsize=8)

    # --- 3. lambda_phys schedule -------------------------------------------
    ax = axes[2]
    ax.plot(epochs, history["lambda_phys"], color="#10B981")
    ax.set_xlabel("epoch")
    ax.set_ylabel("lambda_phys")
    ax.set_title("Physics-loss weight schedule")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"saved plot to {save_path}")
    return fig

def LoadLabels(root_train, root_test, featuretimes, device, pinn):
    labels_true = []
    labels_pred = []
    labels_train_true = []
    labels_train_pred = []
    dataset_test = ds.TrajectoryDataset(root_test, False, featuretimes)
    dataset_train = ds.TrajectoryDataset(root_train, False, featuretimes)
    testlen = len(dataset_test.files)
    trainlen = len(dataset_train.files)
    for i in range(testlen):
        features, labels, rho_t, rho_tp1 = dataset_test.__getitem__(i)
        features = features.to(device)
        taus_pred_log, taus_pred, gammas_pred = pinn.forward(features)
        labels_true.append(torch.exp(labels).detach().cpu().numpy())
        labels_pred.append(taus_pred.detach().cpu().numpy())

    for i in range(trainlen):
        features, labels, rho_t, rho_tp1 = dataset_train.__getitem__(i)
        features = features.to(device)
        taus_pred_log, taus_pred, gammas_pred = pinn.forward(features)
        labels_train_true.append(torch.exp(labels).detach().cpu().numpy())
        labels_train_pred.append(taus_pred.detach().cpu().numpy())


    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)
    labels_train_true = np.array(labels_train_true)
    labels_train_pred = np.array(labels_train_pred)

    return labels_true, labels_pred, labels_train_true, labels_train_pred


def PlotCorelations(col, savepath, labels_true, labels_pred, labels_train_true, labels_train_pred):

    vals = np.concatenate([labels_true[:, col], labels_pred[:, col]])
    vals_train_true = np.concatenate([labels_train_true[:, col]])
    vals_train_pred = np.concatenate([labels_train_pred[:, col]])
    max_val = np.sort(vals)[int(len(vals)*9/10)-1]
    min_val = np.sort(vals)[0]
    max_val_train_true = np.sort(vals_train_true)[len(vals_train_true)-1]
    min_val_train_true = np.sort(vals_train_true)[0]
    max_val_train_pred = np.sort(vals_train_pred)[len(vals_train_pred)-1]
    min_val_train_pred = np.sort(vals_train_pred)[0]


    plt.scatter(labels_true[:, col], labels_pred[:, col], color='blue', alpha=0.6)
    plt.scatter(labels_train_true[:, col], labels_train_pred[:, col], color='red', alpha=0.6)
    plt.plot([min(labels_true[:, col]), max(labels_true[:, col])], [min(labels_true[:, col]), max(labels_true[:, col])], '--', color='green')
    plt.xlabel("T true")
    plt.ylabel("T predicted")
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    ax = plt.gca()
    ax.add_patch(Rectangle((min_val_train_true, min_val_train_pred), max_val_train_true-min_val_train_true, max_val_train_pred-min_val_train_pred,
                           fill=False, edgecolor='red', linewidth=2))

    if savepath != "":
        plt.savefig(Path(savepath) / ("TruePredCorelation" + str(col)), dpi=150)

    plt.show()


def PlotErrorColor(x_cols, y_cols, error_cols, savepath, labels_true, labels_pred, labels_train_true, labels_train_pred):

    xaxis = labels_true[:, x_cols].mean(axis=1)
    yaxis = labels_true[:, y_cols].mean(axis=1)

    xmin = np.min(labels_train_true[:, x_cols].mean(axis=1))
    ymin = np.min(labels_train_true[:, y_cols].mean(axis=1))
    xmax = np.max(labels_train_true[:, x_cols].mean(axis=1))
    ymax = np.max(labels_train_true[:, y_cols].mean(axis=1))

    error = np.mean((labels_pred[:, error_cols] - labels_true[:, error_cols]) ** 2, axis=1)

    plt.scatter(xaxis, yaxis, c=error, cmap='viridis', s=10, norm=LogNorm())
    plt.colorbar(label='error')
    ax = plt.gca()
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                           fill=False, edgecolor='red', linewidth=2))

    if savepath != "":
        x_cols_str = "_".join(map(str, x_cols))
        y_cols_str = "_".join(map(str, y_cols))
        error_cols_str = "_".join(map(str, error_cols))
        plt.savefig(Path(savepath) / ("ErrorColorPlot_x_" + x_cols_str + "_y_" + y_cols_str + "_error_" + error_cols_str), dpi=150)

    plt.show()

    grid_x, grid_y = np.mgrid[
                     xaxis.min():xaxis.max():200j,
                     yaxis.min():yaxis.max():200j
                     ]

    grid_z = griddata((xaxis, yaxis), error, (grid_x, grid_y), method='linear')
    grid_z = gaussian_filter(grid_z, sigma=3)

    plt.imshow(
        grid_z.T,
        origin='lower',
        extent=(xaxis.min(), xaxis.max(), yaxis.min(), yaxis.max()),
        cmap='viridis',
        aspect='equal',
        norm=LogNorm()
    )
    ax = plt.gca()
    ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                           fill=False, edgecolor='red', linewidth=2))
    plt.colorbar(label='Error')

    if savepath != "":
        x_cols_str = "_".join(map(str, x_cols))
        y_cols_str = "_".join(map(str, y_cols))
        error_cols_str = "_".join(map(str, error_cols))
        plt.savefig(Path(savepath) / ("ErrorColorPlotBlurred_x_" + x_cols_str + "_y_" + y_cols_str + "_error_" + error_cols_str), dpi=150)

    plt.show()

