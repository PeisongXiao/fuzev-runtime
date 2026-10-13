"""Load and serve a decision head using the bundle's declared geometry."""

import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from ._scoring import SharedDecisionHead
from .contracts import DEFAULT_HEAD_ID


def read_artifacts(directory):
    directory = Path(directory)
    config = (directory / "config.json").read_bytes()
    metadata = json.loads(config)
    for key in ("hidden_size", "projection_size", "query_adapter_size", "key_adapter_size",
                "primitives", "max_options", "encoding_version"):
        if key not in metadata:
            raise ValueError(f"Head config is missing {key}")
    calibration = json.loads((directory / "calibration.json").read_text())
    temperatures = calibration["temperatures"]
    if (set(temperatures) != set(metadata["primitives"])
            or any(type(t) not in (float, int) or not math.isfinite(t) or t <= 0
                   for t in temperatures.values())):
        raise ValueError("Expected a positive finite temperature for each declared primitive")
    state = load_file(directory / "head.safetensors", device="cpu")
    return metadata, state, temperatures


class ServingHead:
    """One startup-loaded scorer that needs only Torch and safetensors."""

    def __init__(self, directory, *, device=None, calibration="raw", head_id=DEFAULT_HEAD_ID):
        if calibration not in ("raw", "calibrated"):
            raise ValueError("Calibration must be raw or calibrated")
        target = torch.device(device if device is not None else "cuda" if torch.cuda.is_available() else "cpu")
        metadata, state, temperatures = read_artifacts(directory)
        # Allocate geometry only; strict loading supplies every trained parameter.
        with torch.device("meta"):
            bundle = SharedDecisionHead(
                hidden_size=metadata["hidden_size"],
                projection_size=metadata["projection_size"],
                query_adapter_size=metadata["query_adapter_size"],
                key_adapter_size=metadata["key_adapter_size"],
            )
        bundle.load_state_dict(state, strict=True, assign=True)
        self.bundle = bundle.to(device=target, dtype=torch.float32).eval().requires_grad_(False)
        self.metadata = metadata
        self.primitives = tuple(metadata["primitives"])
        self.max_options = metadata["max_options"]
        self.encoding_version = metadata["encoding_version"]
        self.temperatures = temperatures
        self.device = target
        self.head_id = head_id
        self.calibrated = calibration == "calibrated"
        self.calibration_id = calibration

    @torch.inference_mode()
    def score(self, primitive, query, candidates, *, baseline):
        q = query.detach().to(device=self.device, dtype=torch.float32).unsqueeze(0)
        k = candidates.detach().to(device=self.device, dtype=torch.float32).unsqueeze(0)
        b = baseline.detach().to(device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.autocast(device_type=self.device.type, enabled=False):
            logits = self.bundle(primitive, q, k, baseline=b)[0]
            raw = torch.softmax(logits, dim=0)
            calibrated = torch.softmax(logits / self.temperatures[primitive], dim=0) if self.calibrated else raw
        return tuple(torch.stack((logits, raw, calibrated)).cpu().tolist())

    def close(self):
        self.bundle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
