import torch
import numpy as np

"""
Code credits: Ziemowit Olinkiewicz and Jagoda Zawisz
"""

hbar = 1.054571817 * 1e-34
eV = 1.6 * 1e-19

def FixRho(rho):
    '''
    Fixing numerical errors in the density matrix after Runge-Kutta integration.
    :param rho: density matrix
    :return:
        fixed density matrix
    '''

    squeeze = rho.ndim == 2     # Handle both 2D (N,N) and 3D batched (B,N,N) inputs
    if squeeze:
        rho = rho.unsqueeze(0) # if there is a single matrix in the batch, add an additional dim

    ### Corrections

    # Hermiticity
    rho_dag = rho.conj().transpose(1, 2)
    rho = (rho + rho_dag) / 2

    # Add tiny regularization to diagonal to help convergence
    eps = 1e-10
    eye = torch.eye(rho.shape[-1], dtype=rho.dtype, device=rho.device)
    rho = rho + eps * eye.unsqueeze(0)

    # Batched eigen-decomposition
    eigvals, eigvecs = torch.linalg.eigh(rho)

    # Clip negative eigenvalues
    eigvals = torch.clamp(eigvals, min=0)

    # Reconstruct:  V @ diag(λ) @ V†
    rho_fixed = eigvecs @ torch.diag_embed(eigvals.to(eigvecs.dtype)) @ eigvecs.conj().transpose(1, 2)

    # Normalize trace
    trace = rho_fixed.diagonal(dim1=1, dim2=2).sum(dim=1)
    rho_fixed = rho_fixed / trace[:, None, None]

    return rho_fixed.squeeze(0) if squeeze else rho_fixed


class Lindblad_solver:
    """
    Solves the Lindblad master equation for the density matrix rho:
        drho/dt = -i/hbar * [H, rho] + sum_k gamma_k * D[L_k](rho)

    where D[L](rho) = L rho L^dagger - 1/2 * {L^dagger L, rho}  (dissipator)

    H and the jump operators L_k are converted to torch tensors so that the solver can run on GPU/CPU and integrate with autograd if needed.
    """

    def __init__(self, H, L_ops, gammas, dt):
        """
        :param:
            H: (N, N) complex array-like Hamiltonian
            L_ops: list of (N, N) complex array-like jump (Lindblad) operators L_k describing the dissipative channels.
            gammas : list of float, decay rate gamma_k associated with each jump operator L_k.
            dt     : float, integration time step.
        """

        # Store H and each jump operator as complex64 torch tensors.
        self.H      = torch.tensor(np.array(H, dtype=np.complex64))
        self.L_ops  = [torch.tensor(np.array(L, dtype=np.complex64)) for L in L_ops]
        self.gammas = list(gammas)
        self.dt     = dt

        # Precompute L_k^dagger @ L_k for each jump operator
        self.LdagL  = [L.conj().mT @ L for L in self.L_ops]

    def lindblad_rhs(self, rho):
        """
        Evaluate drho/dt at the given state rho, using self.gammas.
        """

        # Match H's device/dtype to rho's in case rho lives on GPU or has a different (e.g. complex128) dtype.
        H = self.H.to(device=rho.device, dtype=rho.dtype)

        # Unitary evolution term: -i/hbar [H, rho]
        drho = -1j / hbar * (H @ rho - rho @ H)

        # Add each dissipator: gamma_k * (L rho L^dagger - 1/2 {L^dagger L, rho})
        for gamma, Lk, LdLk in zip(self.gammas, self.L_ops, self.LdagL):
            Lk = Lk.to(device=rho.device, dtype=rho.dtype)
            LdLk = LdLk.to(device=rho.device, dtype=rho.dtype)
            drho += gamma * (Lk @ rho @ Lk.conj().mT - 0.5 * (LdLk @ rho + rho @ LdLk))

        return drho

    def _renormalize(self, rho):
        """
        Rescale rho so that Tr(rho) = 1.
        """
        return rho / np.trace(rho)

    def _renormalize_batched(self, rho):
        # Normalize trace
        trace = rho.diagonal(dim1=1, dim2=2).sum(dim=1)
        rho = rho / trace[:, None, None]
        return rho

    def step(self, rho):
        """
        Advance rho by one time step dt using classical 4th-order Runge-Kutta, with self.gammas.
        :return:
            rho_next : (N, N) tensor, the density matrix at time t + dt (after being passed through FixRho to enforce physicality)
        """
        dt = self.dt
        L  = self.lindblad_rhs  # shorthand for the RHS function

        # Standard RK4 stages
        k1 = L(rho)
        k2 = L(rho + 0.5 * dt * k1)
        k3 = L(rho + 0.5 * dt * k2)
        k4 = L(rho +       dt * k3)

        # Weighted combination of stages to get the next state
        rho_next = rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # Clean up the result (e.g. re-Hermitize, fix trace/eigenvalues)
        return FixRho(rho_next)

    def run_simulation(self, rho0, n_steps=1000):
        """
        Integrate the Lindblad equation forward in time.

        :param:
            rho0: (N, N) array-like, initial density matrix.
            n_steps : int, number of RK4 steps to take
        :return:
            times : array of shape (n_steps+1,) with the simulation time at each step.
            rhos  : array of shape (n_steps+1, N, N) with the density matrix trajectory.
        """
        rho = torch.tensor(np.array(rho0, dtype=complex))

        # Store the trajectory; start with the initial state at t = 0.
        rhos = [rho.clone()]
        times = [0]

        for i in range(n_steps):
            rho = self.step(rho)
            rhos.append(rho.clone())
            times.append((i + 1) * self.dt)

        return [np.array(times), np.array(rhos)]

    def populations(rhos):
        """
        Extract level populations (diagonal entries) from a trajectory.

        :param:
            rhos: array of shape (T, N, N)

        :return:
            array of shape (T, N) with the real-valued population of each basis state at every time step.
        """
        return np.real(np.einsum('tii->ti', rhos))


    # Batched (per-sample gammas) -- for PINN physics-loss training.
    # H and L_ops stay fixed/shared; only gammas vary per batch element.
    def lindblad_rhs_batched(self, rho, gammas):
        """
        Evaluate drho/dt for a batch of states, each with its own gammas.

        :param
            rho:    (B, N, N) complex tensor
            gammas: (B, K) real tensor, K = number of jump operators (5 here).
                        gammas[:, k] is the rate for L_ops[k], one value per batch element.
        :return:
            drho: (B, N, N) complex tensor
        """
        H = self.H.to(device=rho.device, dtype=rho.dtype)

        # Unitary term: H is identical for every sample, broadcasts over the batch.
        drho = -1j / hbar * (H @ rho - rho @ H)

        for k, (Lk, LdLk) in enumerate(zip(self.L_ops, self.LdagL)):
            Lk = Lk.to(device=rho.device, dtype=rho.dtype)
            LdLk = LdLk.to(device=rho.device, dtype=rho.dtype)

            dissipator = Lk @ rho @ Lk.conj().mT - 0.5 * (LdLk @ rho + rho @ LdLk)

            # gammas[:, k]: (B,) -> (B,1,1) so it broadcasts against (B,N,N)
            gamma_k = gammas[:, k].view(-1, 1, 1).to(device=rho.device, dtype=rho.dtype)
            drho = drho + gamma_k * dissipator

        return drho

    def step_batched(self, rho, gammas):
        """
        Advance a batch of density matrices by one RK4 step of size self.dt,
        each sample using its own (e.g. NN-predicted) gammas.

        :param
            rho: (B, N, N) complex tensor, current states
            gammas: (B, K) real tensor, per-sample decay/dephasing rates
        :return:
            rho_next: (B, N, N) complex tensor, states at t + dt (FixRho applied)
        """
        dt = self.dt
        L  = lambda r: self.lindblad_rhs_batched(r, gammas)  # gammas fixed across RK stages

        k1 = L(rho)
        k2 = L(rho + 0.5 * dt * k1)
        k3 = L(rho + 0.5 * dt * k2)
        k4 = L(rho +       dt * k3)

        rho_next = rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return self._renormalize_batched(rho_next)


def convert_rhos_to_numpy(rhos):
    """
    Convert a trajectory of density matrices into a clean numpy array.

    If an extra singleton batch dimension is present (shape (T, 1, N, N)),
    it is squeezed out so the result is always (T, N, N).
    """
    rhos_np = np.array(rhos)

    if rhos_np.ndim == 4:
        rhos_np = rhos_np.squeeze(1)

    return rhos_np