import matplotlib.pyplot as plt

import numpy as np
import torch
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from pathlib import Path


def PlotCorelations(col, savepath, labels_true, labels_pred):

    par_names = {
        0: "Rel1",
        1: "Rel2",
        2: "Dec1",
        3: "Dec2",
        4: "CorDec"
    }

    parameter = par_names[col]

    vals = np.concatenate([labels_true[:, col], labels_pred[:, col]])
    max_val = np.sort(vals)[int(len(vals)*9/10)-1]
    min_val = np.sort(vals)[0]

    plt.figure()

    plt.scatter(labels_true[:, col], labels_pred[:, col], color='blue', alpha=0.6)
    plt.plot([min(labels_true[:, col]), max(labels_true[:, col])], [min(labels_true[:, col]), max(labels_true[:, col])], '--', color='green')
    plt.xlabel("T true")
    plt.ylabel("T predicted")
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    #os.path.join(out_path, "data.json"
    plt.savefig(Path(savepath) / ("TruePredCorelation_" + parameter), dpi=150)

    #plt.show()


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

