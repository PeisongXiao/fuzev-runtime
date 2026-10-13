"""Shared decision-head math, with geometry supplied by the head bundle."""

import math

import torch
from torch import nn


class CandidateHead(nn.Module):
    def __init__(self, hidden_size, projection_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, projection_size)
        self.key = nn.Linear(hidden_size, projection_size)


class SharedDecisionHead(nn.Module):
    def __init__(self, *, hidden_size, projection_size, query_adapter_size, key_adapter_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.projection_size = projection_size
        self.scorer = CandidateHead(hidden_size, projection_size)
        self.gate = nn.Linear(hidden_size, 1)
        self.adapter_down = nn.Linear(hidden_size, query_adapter_size)
        self.adapter_up = nn.Linear(query_adapter_size, projection_size, bias=False)
        self.channel_a = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
        self.channel_b = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
        self.channel_c = nn.Parameter(torch.zeros(projection_size, dtype=torch.float32))
        self.key_adapter_down = nn.Linear(hidden_size, key_adapter_size)
        self.key_adapter_up = nn.Linear(key_adapter_size, projection_size, bias=False)

    def forward(self, primitive, query, candidates, mask=None, *, baseline):
        hidden = torch.nn.functional.silu(self.adapter_down(query))
        projected_query = self.scorer.query(query) + self.adapter_up(hidden)
        key_hidden = torch.nn.functional.silu(self.key_adapter_down(candidates))
        projected_keys = self.scorer.key(candidates) + self.key_adapter_up(key_hidden)
        channel_logits = (self.channel_a * projected_query[:, None, :]
                          + self.channel_b * projected_keys + self.channel_c)
        # Preserve the trained FP32 reduction order, including the separate residual.
        base_raw = torch.einsum("bd,bkd->bk", projected_query, projected_keys) / math.sqrt(self.projection_size)
        channel_delta = (projected_query[:, None, :] * projected_keys
                         * (2 * torch.sigmoid(channel_logits) - 1)).sum(dim=-1) / math.sqrt(self.projection_size)
        raw = base_raw + channel_delta
        valid = torch.ones_like(raw, dtype=torch.bool) if mask is None else mask
        counts = valid.sum(dim=1, keepdim=True)
        raw = raw.masked_fill(~valid, 0)
        mean = raw.sum(dim=1, keepdim=True) / counts
        delta = (raw - mean).masked_fill(~valid, 0)
        base = baseline.detach().masked_fill(~valid, 0)
        base_mean = base.sum(dim=1, keepdim=True) / counts
        centered_base = (base - base_mean).masked_fill(~valid, 0)
        gate_logits = self.gate(query)
        trust = 2 * torch.sigmoid(gate_logits)
        logits = base + delta + (trust - 1) * centered_base
        return logits if mask is None else logits.masked_fill(~mask, -torch.inf)
