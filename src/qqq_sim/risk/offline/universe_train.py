"""Deterministic encoder initialization and an actual trainable torch adapter."""
from __future__ import annotations
import numpy as np


def initialize_universe_arrays(seed=20261109):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(9, 0))))
    return {"universe_w0": rng.standard_normal((16, 64)) * (2 / 16) ** .5,
            "universe_b0": np.zeros(64, dtype=np.float64),
            "universe_w1": rng.standard_normal((64, 64)) * (2 / 64) ** .5,
            "universe_b1": np.zeros(64, dtype=np.float64)}


def make_trainable_encoder(arrays=None):
    import torch
    from torch import nn
    arrays = initialize_universe_arrays() if arrays is None else arrays
    class UniverseEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = nn.Linear(16, 64, dtype=torch.float64)
            self.second = nn.Linear(64, 64, dtype=torch.float64)
            with torch.no_grad():
                self.first.weight.copy_(torch.as_tensor(arrays["universe_w0"].T))
                self.first.bias.copy_(torch.as_tensor(arrays["universe_b0"]))
                self.second.weight.copy_(torch.as_tensor(arrays["universe_w1"].T))
                self.second.bias.copy_(torch.as_tensor(arrays["universe_b1"]))

        def forward(self, tokens):
            mask = tokens[..., 0:1]
            count = mask.sum(dim=-2)
            hidden = nn.functional.silu(self.second(nn.functional.silu(self.first(tokens))))
            total = (hidden * mask).sum(dim=-2) / torch.sqrt(torch.clamp(count, min=1))
            maximum = hidden.masked_fill(mask == 0, -torch.inf).max(dim=-2).values
            maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
            summaries = []
            expiry_count = torch.zeros_like(count)
            for ordinal in range(1, 9):
                selected = (torch.round(tokens[..., 15:16] * 8) == ordinal) * mask
                per_expiry = selected.sum(dim=-2)
                present = (per_expiry > 0).to(tokens.dtype)
                remaining = (tokens[..., 2:3] * selected).sum(dim=-2) / torch.clamp(per_expiry, min=1)
                summaries.extend((present, remaining, per_expiry / 512))
                expiry_count = expiry_count + present
            return torch.cat((total, maximum, count / 512, expiry_count / 8, *summaries), dim=-1)

        def export_arrays(self):
            return {"universe_w0": self.first.weight.detach().cpu().numpy().T.copy(),
                    "universe_b0": self.first.bias.detach().cpu().numpy().copy(),
                    "universe_w1": self.second.weight.detach().cpu().numpy().T.copy(),
                    "universe_b1": self.second.bias.detach().cpu().numpy().copy()}
    return UniverseEncoder()
