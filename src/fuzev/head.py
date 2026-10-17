"""Load and serve a decision head using the bundle's declared geometry."""

import json
import logging
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from ._scoring import SharedDecisionHead
from .contracts import DEFAULT_HEAD_ID, Primitive
from .encoding import LIST_AWARE_VERSION


V2_SCHEMA = "fuzev-shared-head-package-v2"
logger = logging.getLogger(__name__)


def read_v2(metadata, calibration):
    if metadata["encoding_version"] != LIST_AWARE_VERSION:
        raise ValueError("Unsupported v2 head encoding")
    for key in ("hidden_size", "projection_size", "query_adapter_size", "key_adapter_size"):
        if type(metadata[key]) is not int or metadata[key] <= 0:
            raise ValueError(f"Expected positive integer head geometry: {key}")
    capacities = metadata.get("capacities")
    if not isinstance(capacities, dict) or not capacities:
        raise ValueError("Expected per-primitive v2 capacities")
    modalities = set()
    for primitive, supported in capacities.items():
        if primitive not in tuple(Primitive) or not isinstance(supported, dict) or not supported:
            raise ValueError("Unsupported primitive or empty modality capacities")
        for modality, capacity in supported.items():
            if modality not in ("text", "image_text") or not isinstance(capacity, dict):
                raise ValueError("Unsupported head modality or capacity")
            maximum = capacity.get("implemented_max")
            if (type(maximum) is not int or maximum < 2
                    or primitive == "noul" and maximum != 2):
                raise ValueError("Expected an integer implemented_max of at least 2, and 2 for noul")
            modalities.add(modality)
    strengths = metadata.get("residual_strength")
    if (not isinstance(strengths, dict) or set(strengths) != modalities
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in strengths.values())):
        raise ValueError("Expected a finite residual strength for each supported modality")
    strata = calibration.get("strata")
    if not isinstance(strata, list):
        raise ValueError("Expected v2 calibration strata")
    intervals = {}
    for row in strata:
        if not isinstance(row, dict):
            raise ValueError("Invalid calibration stratum")
        primitive, modality = row.get("primitive"), row.get("modality")
        if not isinstance(primitive, str) or not isinstance(modality, str):
            raise ValueError("Invalid calibration primitive or modality")
        capacity = capacities.get(primitive, {}).get(modality)
        low, high, temperature = row.get("k_min"), row.get("k_max"), row.get("temperature")
        if (capacity is None or type(low) is not int or type(high) is not int
                or not 2 <= low <= high <= capacity["implemented_max"]
                or type(temperature) not in (float, int) or not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("Invalid calibration stratum bounds or temperature")
        intervals.setdefault((primitive, modality), []).append((low, high))
    for spans in intervals.values():
        previous = 1
        for low, high in sorted(spans):
            if low <= previous:
                raise ValueError("Overlapping calibration strata")
            previous = high
    return strata


def calibration_temperature(strata, primitive, *, modality="text", count):
    """Return a declared temperature, including 1.0, or None for an uncovered request."""
    return next((float(row["temperature"]) for row in strata
                 if row["primitive"] == primitive and row["modality"] == modality
                 and row["k_min"] <= count <= row["k_max"]), None)


def read_artifacts(directory):
    directory = Path(directory)
    config = (directory / "config.json").read_bytes()
    metadata = json.loads(config)
    if metadata.get("schema") not in (None, "fuzev-shared-head-package-v1", V2_SCHEMA):
        raise ValueError("Unsupported head bundle schema")
    v2 = metadata.get("schema") == V2_SCHEMA
    required = ("hidden_size", "projection_size", "query_adapter_size", "key_adapter_size", "encoding_version")
    for key in required + (() if v2 else ("primitives", "max_options")):
        if key not in metadata:
            raise ValueError(f"Head config is missing {key}")
    if not v2 and metadata["encoding_version"] == LIST_AWARE_VERSION:
        raise ValueError("The list-aware v2 encoding requires a v2 head bundle")
    calibration = json.loads((directory / "calibration.json").read_text())
    if v2:
        temperatures = read_v2(metadata, calibration)
    else:
        temperatures = calibration["temperatures"]
        if (set(temperatures) != set(metadata["primitives"])
                or any(type(t) not in (float, int) or not math.isfinite(t) or t <= 0
                       for t in temperatures.values())):
            raise ValueError("Expected a positive finite temperature for each declared primitive")
    state = load_file(directory / "head.safetensors", device="cpu")
    return metadata, state, temperatures


class ServingHead:
    """One startup-loaded scorer that needs only Torch and safetensors."""

    def __init__(self, directory, *, device=None, calibration="raw"):
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
        self.is_v2 = metadata.get("schema") == V2_SCHEMA
        self.capacities = metadata["capacities"] if self.is_v2 else None
        self.primitives = tuple(self.capacities if self.is_v2 else metadata["primitives"])
        self.max_options = (max(capacity["implemented_max"] for supported in self.capacities.values()
                                for capacity in supported.values()) if self.is_v2 else metadata["max_options"])
        self.encoding_version = metadata["encoding_version"]
        self.temperatures = temperatures
        self.device = target
        self.head_id = metadata.get("head_id", DEFAULT_HEAD_ID)
        self.calibrated = calibration == "calibrated"
        self.calibration_id = calibration
        self._warned_uncalibrated = False

    def temperature_for(self, primitive, *, modality="text", count):
        """Return the temperature for this request, or None when uncalibrated or uncovered."""
        if not self.calibrated:
            return None
        if not self.is_v2:
            return float(self.temperatures[primitive])
        return calibration_temperature(self.temperatures, primitive, modality=modality, count=count)

    @torch.inference_mode()
    def score(self, primitive, query, candidates, *, baseline, modality="text"):
        q = query.detach().to(device=self.device, dtype=torch.float32).unsqueeze(0)
        k = candidates.detach().to(device=self.device, dtype=torch.float32).unsqueeze(0)
        b = baseline.detach().to(device=self.device, dtype=torch.float32).unsqueeze(0)
        with torch.autocast(device_type=self.device.type, enabled=False):
            strength = self.metadata["residual_strength"][modality] if self.is_v2 else 1
            if strength == 0:
                logits = b[0]
            else:
                logits = self.bundle(primitive, q, k, baseline=b)[0]
                if strength != 1:
                    logits = b[0] + strength * (logits - b[0])
            raw = torch.softmax(logits, dim=0)
            temperature = self.temperature_for(primitive, modality=modality, count=len(candidates))
            if temperature is not None:
                calibrated = torch.softmax(logits / temperature, dim=0)
            else:
                calibrated = raw
                if self.calibrated and not self._warned_uncalibrated:
                    self._warned_uncalibrated = True
                    logger.warning("No calibration stratum for %s/%s K=%s; uncovered requests use raw probabilities",
                                   primitive, modality, len(candidates))
        return tuple(torch.stack((logits, raw, calibrated)).cpu().tolist())

    def close(self):
        self.bundle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
