"""Differentiable five-channel Lindblad model used only by the physics loss."""
import math
import torch
from config import HBAR_MEV_PS, PSI_NP


def build_system(device, J_meV=0.0):
    cd = torch.complex64
    I = torch.eye(2, dtype=cd, device=device)
    Z = torch.tensor([[1, 0], [0, -1]], dtype=cd, device=device)
    SM = torch.tensor([[0, 1], [0, 0]], dtype=cd, device=device)

    Lops = torch.stack((
        torch.kron(SM, I),
        torch.kron(I, SM),
        torch.kron(Z, I),
        torch.kron(I, Z),
        torch.kron(Z, I) + torch.kron(I, Z),
    ))

    Hs = torch.tensor([1., 0.], dtype=cd, device=device)
    Vs = torch.tensor([0., 1.], dtype=cd, device=device)
    Ds = torch.tensor([1., 1.], dtype=cd, device=device) / math.sqrt(2)
    Rs = torch.tensor([1., -1j], dtype=cd, device=device) / math.sqrt(2)

    projectors = []
    for a in (Hs, Vs, Ds, Rs):
        for b in (Hs, Vs, Ds, Rs):
            q = torch.kron(a, b)
            projectors.append(torch.outer(q, q.conj()))
    projectors = torch.stack(projectors)

    psi = torch.tensor(PSI_NP, dtype=cd, device=device)
    psi /= torch.linalg.vector_norm(psi)
    rho0 = torch.outer(psi, psi.conj())

    M = torch.tensor(
        [[1, 0, 0, 0], [0, -1, 2, 0], [0, 2, -1, 0], [0, 0, 0, 1]],
        dtype=cd, device=device,
    )
    H = (J_meV / (4 * HBAR_MEV_PS)) * M
    return H, Lops, projectors, rho0


def theta_to_gamma(theta):
    """
    Cache convention:
      gamma_T1 = 1/T1
      gamma_Z  = 1/(2 Tphi), because D[Z] damps coherence at 2 gamma
      gamma_corr = 1/tau_corr
    """
    g = torch.empty_like(theta)
    g[:, 0] = 1 / theta[:, 0]
    g[:, 1] = 1 / theta[:, 1]
    g[:, 2] = 1 / (2 * theta[:, 2])
    g[:, 3] = 1 / (2 * theta[:, 3])
    g[:, 4] = 1 / theta[:, 4]
    return g


def lindblad_rhs(rho, gamma, H, Lops):
    out = -1j * (H @ rho - rho @ H)
    for k, L in enumerate(Lops):
        Ld = L.conj().T
        LdL = Ld @ L
        dissipator = L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL)
        out = out + gamma[:, k, None, None] * dissipator
    return out


def evolve(theta, t, H, Lops, rho0, max_step=0.1):
    rho = rho0[None].expand(len(theta), -1, -1).clone()
    gamma = theta_to_gamma(theta)
    nsteps = max(1, int(math.ceil(float(t) / max_step)))
    h = float(t) / nsteps

    for _ in range(nsteps):
        k1 = lindblad_rhs(rho, gamma, H, Lops)
        k2 = lindblad_rhs(rho + 0.5 * h * k1, gamma, H, Lops)
        k3 = lindblad_rhs(rho + 0.5 * h * k2, gamma, H, Lops)
        k4 = lindblad_rhs(rho + h * k3, gamma, H, Lops)
        rho = rho + h * (k1 + 2*k2 + 2*k3 + k4) / 6
    return 0.5 * (rho + rho.conj().transpose(-1, -2))


def probabilities(theta, times, system, max_step=0.1):
    H, Lops, P, rho0 = system
    chunks = []
    for t in times:
        rho = evolve(theta, t, H, Lops, rho0, max_step)
        p = torch.einsum("kij,bji->bk", P, rho).real
        chunks.append(p.clamp(1e-6, 1 - 1e-6))
    return torch.cat(chunks, dim=1)


def binomial_physics_nll(theta_pred, n, N, system, times, max_step=0.1):
    p = probabilities(theta_pred, times, system, max_step)
    Nf = N.float()
    ll = n * torch.log(p) + (Nf - n) * torch.log1p(-p)
    return -ll.sum() / Nf.sum().clamp_min(1.0)
