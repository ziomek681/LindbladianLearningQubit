# Adaptive Lindblad learning — clean current version

This folder contains the current project stripped to the pieces that are actually
used by the successful pipeline.

## Physics problem

Five unknown parameters:

`[T1_q1, T1_q2, Tphi_q1, Tphi_q2, tau_corr]`

with five jump operators:

1. `sigma_- ⊗ I`
2. `I ⊗ sigma_-`
3. `Z ⊗ I`
4. `I ⊗ Z`
5. `Z ⊗ I + I ⊗ Z`

The fixed asymmetric initial state is defined in `config.py`.
There are 16 Kwiat projectors at five times, hence 80 settings.

## Current algorithm

Training and deployment are intentionally separated.

**Deployment**

`global Bayesian A-opt warmup`
→ measurements
→ Transformer estimate `theta_hat`
→ accumulated local Fisher A-opt at `theta_hat`
→ more measurements
→ repeat.

The adaptive allocation is safety-mixed with the global Bayesian allocation:

`w = (1-epsilon) w_local + epsilon w_Bayes`, default `epsilon=0.2`.

The local design includes **all accumulated Fisher information**, not only the
new batch.

**Training**

Default mixture is now:

- 10% uniform
- 20% static Bayesian
- 70% closed-loop adaptive

The decoder is trained on the same allocation family it sees at deployment.

Loss:

`L = L_supervised + lambda_phys L_physics`

where `L_physics` is the differentiable binomial negative log-likelihood obtained
from the Lindblad model evaluated at the predicted parameters.

**Oracle**

Oracle is never used for training. It is only a benchmark: the local Fisher
tensor is evaluated at `theta_true` rather than `theta_hat`.

The final benchmark reports both:

- Transformer RMSE for Uniform / Bayes / Adaptive / Oracle
- true-Fisher A-score of the actual allocations

The information-theoretic diagnostic should show approximately

`Oracle < Adaptive < Bayes < Uniform`

for A-score (lower is better).

## Files

- `config.py` — parameter ranges, initial state, times and budgets.
- `model.py` — Transformer and parameter normalization.
- `decoder.py` — physical-parameter decoder wrapper.
- `measurements.py` — shot sampling, integer allocations, token features.
- `design.py` — Bayesian A-opt, accumulated local Fisher, adaptive/oracle loops.
- `physics.py` — differentiable Lindblad propagation and physics NLL.
- `io_utils.py` — cache loading, RMSE and true-Fisher diagnostics.
- `train.py` — training, validation and final benchmark only.

## Required cached data

Place next to the scripts:

- `lindblad_probability_pool.pt`
- `oracle_fisher_settings_pool.pt`

The Fisher cache and probability cache must correspond to the same theta pool
and the same five measurement times.

## Run

Current long run:

```bash
python train.py
```

Equivalent explicit important settings:

```bash
python train.py \
  --epochs 150 \
  --steps-per-epoch 200 \
  --p-uniform 0.10 \
  --p-bayes 0.20 \
  --lambda-phys 0.20 \
  --n-test 512
```

Quick smoke test:

```bash
python train.py --epochs 2 --steps-per-epoch 4 --n-test 32
```

## Conventions that should not be changed accidentally

For `L=Z`, `gamma D[Z]` damps coherence at rate `2 gamma`, therefore

`gamma_Z = 1/(2 Tphi)`.

Relaxation uses `gamma_T1 = 1/T1`.

The correlated channel currently uses `gamma_corr = 1/tau_corr`, matching the
existing probability/Fisher caches.

The physics integrator uses the supplied `max_step`; there is no hard-coded
time step.
