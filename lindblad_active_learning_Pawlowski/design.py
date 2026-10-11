"""Global Bayesian and sequential local Fisher A-optimal design."""
import torch
from model import theta_to_z
from measurements import weights_to_counts, sample_counts, equal_counts, features_from_counts


@torch.no_grad()
def bayesian_aopt_weights(Ftrain, K, prior_precision=3.0, iters=32,
                          explore=0.10, chunk=512):
    """
    Monte-Carlo Bayesian A-optimal design:
        min_w E_theta Tr[(Lambda + K sum_m w_m F_m(theta))^-1].
    """
    M, S = Ftrain.shape[:2]
    device = Ftrain.device
    w = torch.full((S,), 1.0 / S, device=device)
    I = torch.eye(5, device=device)

    for _ in range(iters):
        gain = torch.zeros(S, device=device)
        for lo in range(0, M, chunk):
            Fc = Ftrain[lo:lo + chunk]
            Ftot = prior_precision * I[None]
            Ftot = Ftot + K * (w[None, :, None, None] * Fc).sum(dim=1)
            Finv = torch.linalg.pinv(Ftot, rcond=1e-6)
            A = Finv[:, None] @ Fc @ Finv[:, None]
            gain += torch.diagonal(A, dim1=-2, dim2=-1).sum(-1).sum(0)

        gain = (gain / M).clamp_min(1e-20)
        w *= torch.sqrt(gain / gain.mean())
        w /= w.sum()

    w = (1.0 - explore) * w + explore / S
    return w / w.sum()


@torch.no_grad()
def sequential_local_aopt_weights(Fset, Nold, delta_K,
                                  prior_precision=3.0, iters=32):
    """
    Local A-optimal increment conditioned on all previously accumulated shots.

    F_old = Lambda + sum_m Nold_m F_m
    choose w for:
        F_new = F_old + delta_K sum_m w_m F_m
    """
    B, S = Fset.shape[:2]
    device = Fset.device
    I = torch.eye(5, device=device, dtype=Fset.dtype)[None]
    Fold = prior_precision * I + (
        Nold.float()[:, :, None, None] * Fset
    ).sum(dim=1)

    w = torch.full((B, S), 1.0 / S, device=device, dtype=Fset.dtype)
    for _ in range(iters):
        Ftot = Fold + delta_K * (
            w[:, :, None, None] * Fset
        ).sum(dim=1)
        Finv = torch.linalg.pinv(Ftot, rcond=1e-6)
        A = Finv[:, None] @ Fset @ Finv[:, None]
        gain = torch.diagonal(A, dim1=-2, dim2=-1).sum(-1).clamp_min(1e-20)
        w *= torch.sqrt(gain / gain.mean(dim=1, keepdim=True))
        w /= w.sum(dim=1, keepdim=True)
    return w


@torch.no_grad()
def nearest_fisher_cpu(theta_hat, theta_train, Ftrain):
    """Cached local Fisher tensor nearest to theta_hat in normalized coordinates."""
    z_hat = theta_to_z(theta_hat).cpu()
    z_train = theta_to_z(theta_train).cpu()
    out = []
    for z in z_hat:
        j = torch.argmin(((z_train - z) ** 2).sum(dim=1))
        out.append(Ftrain[j])
    return torch.stack(out)

@torch.no_grad()
def nearest_fisher(theta_hat, theta_train, Ftrain):
    """Find nearest training point using GPU vectorized distances."""
    z_hat = theta_to_z(theta_hat)
    z_train = theta_to_z(theta_train)

    # Squared Euclidean distances: [B, N]
    dist2 = (
        z_hat.square().sum(dim=1, keepdim=True)
        + z_train.square().sum(dim=1).unsqueeze(0)
        - 2.0 * (z_hat @ z_train.T)
    )

    idx = dist2.argmin(dim=1)
    return Ftrain[idx]


@torch.no_grad()
def static_batch(p, K, mode, bayes_counts, device):
    B, S = p.shape
    if mode == "uniform":
        N = equal_counts(B, S, K, device)
    elif mode == "bayes":
        N = bayes_counts[K].to(device)[None].repeat(B, 1)
    else:
        raise ValueError(mode)
    return sample_counts(p, N), N


@torch.no_grad()
def sequential_batch(model, p, target_K, theta_train, Ftrain,
                     bayes_counts, bayes_weights, p_mean, p_std,
                     warmup, adaptive_Ks, epsilon, prior_precision,
                     design_iters, device, Ftrue=None):
    """
    Closed-loop experiment.

    Ftrue=None  -> adaptive design using F(theta_hat)
    Ftrue!=None -> oracle diagnostic using F(theta_true)

    Both start from the same global Bayesian warmup and use accumulated Fisher.
    """
    B, S = p.shape
    warm = min(warmup, target_K)
    Ntot = equal_counts(B, S, warm, device).clone()
    #Ntot = bayes_counts[warm].to(device)[None].repeat(B, 1).clone()
    ntot = sample_counts(p, Ntot)

    if target_K <= warm:
        return ntot, Ntot

    def local_fisher(current_K):
        if Ftrue is not None:
            return Ftrue.to(device)
        x = features_from_counts(ntot, Ntot, current_K, p_mean, p_std)
        theta_hat = model.theta_from_features(x)
        return nearest_fisher(theta_hat, theta_train, Ftrain)
        #return nearest_fisher_cpu(theta_hat.cpu(), theta_train, Ftrain).to(device)

    previous_K = warm
    checkpoints = [K for K in adaptive_Ks if warm < K <= target_K]
    if target_K not in checkpoints:
        checkpoints.append(target_K)

    for K in checkpoints:
        delta_K = K - previous_K
        Flocal = local_fisher(previous_K)
        w_local = sequential_local_aopt_weights(
            Flocal, Ntot, delta_K, prior_precision, design_iters
        )
        w_bayes = bayes_weights[K].to(device)[None].repeat(B, 1)
        w = (1.0 - epsilon) * w_local + epsilon * w_bayes
        w /= w.sum(dim=1, keepdim=True)

        Nnew = weights_to_counts(w, delta_K).to(device)
        ntot += sample_counts(p, Nnew)
        Ntot += Nnew
        previous_K = K

    return ntot, Ntot
