"""Shot sampling, integer allocations and Transformer input construction."""
import torch


def sample_counts(p, N):
    return torch.distributions.Binomial(
        total_count=N.float(), probs=p
    ).sample()


def equal_counts(batch_size, n_settings, K, device):
    N = torch.full(
        (batch_size, n_settings), K // n_settings,
        dtype=torch.long, device=device,
    )
    if K % n_settings:
        N[:, : K % n_settings] += 1
    return N


@torch.no_grad()
def weights_to_counts(w, K):
    """Largest-remainder rounding; each row sums exactly to K."""
    if w.ndim == 1:
        w = w[None]
    raw = w * K
    N = torch.floor(raw).long()
    remainder = K - N.sum(dim=1)
    fractional = raw - N.float()
    order = torch.argsort(fractional, dim=1, descending=True)
    for b in range(len(N)):
        r = int(remainder[b])
        if r:
            N[b, order[b, :r]] += 1
    return N


def features_from_counts(n, N, K, p_mean, p_std):
    """
    Token features:
      0: standardized empirical probability
      1: allocation fraction N_m / K
    """
    Nf = N.float()
    phat = torch.where(N > 0, n / Nf.clamp_min(1), torch.zeros_like(n))
    pz = (phat - p_mean.to(n.device)) / p_std.to(n.device)
    pz = torch.where(N > 0, pz, torch.zeros_like(pz))
    return torch.stack((pz, Nf / float(K)), dim=-1)
