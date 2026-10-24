"""Single source of truth for the five-parameter Lindblad problem."""
import torch

SEED = 1234

PARAM_NAMES = ("T1_q1", "T1_q2", "Tphi_q1", "Tphi_q2", "tau_corr")
LOW  = torch.tensor([16., 16., 20., 20.,  50.])
HIGH = torch.tensor([24., 24., 80., 80., 200.])
MID  = 0.5 * (LOW + HIGH)
HALF = 0.5 * (HIGH - LOW)

HBAR_MEV_PS = 0.6582119569
PSI_NP = (
    -0.0245922 - 0.22576801j,
     0.46178448 + 0.25162446j,
    -0.49144100 + 0.13995686j,
    -0.49401122 - 0.40830746j,
)

DEFAULT_TIMES = (0.5, 1.0, 2.0, 5.0, 10.0)
DEFAULT_KS = (4096, 8192, 16384, 32768, 65536, 131072)
DEFAULT_ADAPTIVE_KS = (8192, 16384, 32768, 65536, 131072)
