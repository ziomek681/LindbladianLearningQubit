import matplotlib.pyplot as plt


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