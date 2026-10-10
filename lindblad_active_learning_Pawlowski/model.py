"""Transformer decoder: 80 measurement tokens -> five Lindblad parameters."""
import torch
import torch.nn as nn
from config import MID, HALF


def theta_to_z(theta):
    return (theta - MID.to(theta.device)) / HALF.to(theta.device)


def z_to_theta(z):
    return MID.to(z.device) + HALF.to(z.device) * z


class MeasurementTransformer(nn.Module):
    def __init__(self, n_settings=80, d_model=128, nhead=8,
                 nlayers=3, dim_ff=256, dropout=0.05):
        super().__init__()
        self.value_embed = nn.Sequential(
            nn.Linear(2, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        self.setting_embed = nn.Embedding(n_settings, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, nlayers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.SiLU(),
            nn.Linear(128, 5),
            nn.Tanh(),
        )

    def forward(self, x):
        setting_ids = torch.arange(x.shape[1], device=x.device)
        h = self.value_embed(x) + self.setting_embed(setting_ids)[None]
        return self.head(self.encoder(h).mean(dim=1))
