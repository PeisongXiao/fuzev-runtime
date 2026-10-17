import json
import math
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from fuzev._scoring import SharedDecisionHead
from fuzev.api import models_response
from fuzev.head import ServingHead
from fuzev.encoding import LIST_AWARE_VERSION


@pytest.fixture(params=[(64, 13, 7, 3), (4096, 8, 4, 2)])
def bundle(tmp_path, request):
    width, projection, query_adapter, key_adapter = request.param
    geometry = dict(hidden_size=width, projection_size=projection,
                    query_adapter_size=query_adapter, key_adapter_size=key_adapter)
    with torch.device("meta"):
        model = SharedDecisionHead(**geometry)
    state = {
        name: (((torch.arange(value.numel(), dtype=torch.float32) + index * 3) % 17 - 8)
               .reshape(value.shape) / (10 * math.sqrt(width)))
        for index, (name, value) in enumerate(model.state_dict().items())
    }
    metadata = dict(geometry, primitives=["select", "rank"], max_options=24,
                    encoding_version="example-branches-v2", release_date="2026-10-12")
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    (tmp_path / "calibration.json").write_text(json.dumps({"temperatures": {"select": 1.5, "rank": 0.75}}))
    save_file(state, tmp_path / "head.safetensors")
    return tmp_path, metadata, state


def features(width, count):
    query = ((torch.arange(width, dtype=torch.float32) % 37) - 18) / 37
    candidates = (((torch.arange(count * width, dtype=torch.float32).reshape(count, width) * 7) % 53) - 26) / 53
    baseline = torch.linspace(-1.2, 0.6, count)
    return query, candidates, baseline


def reference_logits(state, query, candidates, baseline):
    def linear(name, value):
        return F.linear(value, state[name + ".weight"], state.get(name + ".bias"))

    q = linear("scorer.query", query) + linear("adapter_up", F.silu(linear("adapter_down", query)))
    k = linear("scorer.key", candidates) + linear("key_adapter_up", F.silu(linear("key_adapter_down", candidates)))
    channels = 2 * torch.sigmoid(state["channel_a"] * q + state["channel_b"] * k + state["channel_c"])
    raw = (q * k * channels).sum(-1) / math.sqrt(q.shape[-1])
    trust = 2 * torch.sigmoid(linear("gate", query))
    return baseline + raw - raw.mean() + (trust - 1) * (baseline - baseline.mean())


@pytest.mark.parametrize("calibration", ["raw", "calibrated"])
def test_bundle_geometry_and_scoring(bundle, calibration):
    directory, metadata, state = bundle
    with ServingHead(directory, device="cpu", calibration=calibration) as head:
        assert head.metadata == metadata
        assert head.max_options == 24
        assert head.primitives == ("select", "rank")
        assert head.encoding_version == "example-branches-v2"
        assert head.bundle.adapter_down.out_features == metadata["query_adapter_size"]
        assert head.bundle.key_adapter_down.out_features == metadata["key_adapter_size"]
        for name, value in head.bundle.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        query, candidates, baseline = features(metadata["hidden_size"], head.max_options)
        baseline.requires_grad_(True)
        expected = reference_logits(state, query, candidates, baseline.detach())
        for primitive in head.primitives:
            result = head.score(primitive, query, candidates, baseline=baseline)
            assert isinstance(result, tuple)
            assert all(isinstance(row, list) and len(row) == head.max_options for row in result)
            temperature = head.temperatures[primitive] if head.calibrated else 1
            assert head.temperature_for(primitive, count=head.max_options) == (
                head.temperatures[primitive] if head.calibrated else None)
            torch.testing.assert_close(torch.tensor(result), torch.stack((
                expected, expected.softmax(0), (expected / temperature).softmax(0),
            )), rtol=2e-6, atol=2e-6)
    assert head.bundle is None
    assert baseline.grad is None


def test_scoring_inside_cpu_autocast(bundle):
    directory, metadata, _ = bundle
    query, candidates, baseline = features(metadata["hidden_size"], 3)
    with ServingHead(directory, device="cpu") as head:
        expected = head.score("select", query, candidates, baseline=baseline)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            actual = head.score("select", query, candidates, baseline=baseline)
    assert actual == expected


@pytest.mark.parametrize("identity", [None, "my-head"])
def test_head_identity_comes_from_bundle(bundle, identity):
    directory, metadata, _ = bundle
    if identity is not None:
        metadata["head_id"] = identity
        (directory / "config.json").write_text(json.dumps(metadata))
    with ServingHead(directory, device="cpu") as head:
        assert head.head_id == (identity or "reference")
        assert models_response("my-model", head)["models"][0]["release_date"] == metadata["release_date"]


def test_masked_batch_matches_independent_scores(bundle):
    directory, metadata, _ = bundle
    query, candidates, baseline = features(metadata["hidden_size"], 4)
    with ServingHead(directory, device="cpu") as head:
        mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
        with torch.inference_mode():
            batched = head.bundle(
                "select", query.expand(2, -1), candidates.expand(2, -1, -1),
                mask=mask, baseline=baseline.expand(2, -1),
            )
        for row, count in enumerate((2, 4)):
            independent = head.score("select", query, candidates[:count], baseline=baseline[:count])[0]
            torch.testing.assert_close(batched[row, :count], torch.tensor(independent), rtol=2e-6, atol=2e-6)
        assert torch.isneginf(batched[0, 2:]).all()


@pytest.mark.parametrize("bundle", [(64, 13, 7, 3)], indirect=True)
def test_unknown_calibration_name(bundle):
    with pytest.raises(ValueError, match="Calibration must be raw or calibrated"):
        ServingHead(bundle[0], device="cpu", calibration="unknown")


@pytest.mark.parametrize("bundle", [(64, 13, 7, 3)], indirect=True)
def test_bundle_temperatures_are_complete_and_positive(bundle):
    directory = bundle[0]
    for temperatures in ({"select": 1.}, {"select": 0., "rank": 1.}):
        (directory / "calibration.json").write_text(json.dumps({"temperatures": temperatures}))
        with pytest.raises(ValueError, match="positive finite temperature"):
            ServingHead(directory, device="cpu")


# Reference outputs of head 1.1.0 on CPU for the inputs from features().
GOLDEN = {
    "choice": [
        [-1.1754971743, -0.5390906334, -0.0080234651, 0.5226111412],
        [0.0864517689, 0.1633660346, 0.2778441608, 0.4723380804],
        [0.1047913879, 0.1789120287, 0.2795763314, 0.4367202520],
    ],
    "noul": [
        [-1.2045214176, 0.6045213342],
        [0.1407538652, 0.8592461348],
        [0.2026261985, 0.7973737717],
    ],
    "ordinal": [
        [-1.1802737713, -0.6904467344, -0.3059591055, 0.0780959874, 0.5985835195],
        [0.0690988228, 0.1127716079, 0.1656458825, 0.2432056367, 0.4092780948],
        [0.0777419433, 0.1210085973, 0.1712585241, 0.2422804534, 0.3877104819],
    ],
}


@pytest.fixture(scope="module")
def published_bundle():
    location = os.environ.get("FUZEV_HEAD_DIRECTORY")
    if not location:
        pytest.skip("Set FUZEV_HEAD_DIRECTORY to the downloaded released head bundle")
    return Path(location)


@pytest.mark.parametrize("primitive", GOLDEN)
@pytest.mark.parametrize("calibration", ["raw", "calibrated"])
def test_published_head_cpu_golden(published_bundle, primitive, calibration):
    with ServingHead(published_bundle, device="cpu", calibration=calibration) as head:
        query, candidates, baseline = features(head.metadata["hidden_size"], len(GOLDEN[primitive][0]))
        actual = head.score(primitive, query, candidates, baseline=baseline)
        expected = torch.tensor(GOLDEN[primitive])
        if calibration == "raw":
            expected[2] = expected[1]
        torch.testing.assert_close(torch.tensor(actual), expected, rtol=2e-6, atol=2e-6)


@pytest.fixture
def v2_bundle(bundle):
    directory, metadata, state = bundle
    metadata = {key: value for key, value in metadata.items() if key not in ("primitives", "max_options")}
    metadata.update(
        schema="fuzev-shared-head-package-v2",
        encoding_version=LIST_AWARE_VERSION,
        capacities={
            "choice": {"text": {"implemented_max": 255, "validated_max": None},
                       "image_text": {"implemented_max": 31}},
            "ordinal": {"text": {"implemented_max": 64, "validated_max": None}},
            "noul": {"text": {"implemented_max": 2, "validated_max": None},
                     "image_text": {"implemented_max": 2, "validated_max": None}},
        }, residual_strength={"text": 1.0, "image_text": 1.0},
    )
    strata = [dict(primitive=primitive, modality=modality, k_min=low, k_max=high, temperature=temperature)
              for primitive, modality, low, high, temperature in (
                  ("choice", "text", 2, 26, 1.25), ("choice", "text", 27, 255, 1.75),
                  ("choice", "image_text", 2, 31, 2.0), ("ordinal", "text", 2, 26, 0.75),
                  ("ordinal", "text", 27, 64, 1.5), ("noul", "text", 2, 2, 0.8),
                  ("noul", "image_text", 2, 2, 2.0))]
    (directory / "config.json").write_text(json.dumps(metadata))
    (directory / "calibration.json").write_text(json.dumps({"strata": strata}))
    return directory, metadata, state


@pytest.mark.parametrize("count", [2, 26, 27, 255])
@pytest.mark.parametrize("calibration", ["raw", "calibrated"])
def test_v2_scores_match_reference_math(v2_bundle, count, calibration):
    directory, metadata, state = v2_bundle
    query, candidates, baseline = features(metadata["hidden_size"], count)
    if count > 26:
        baseline.zero_()
    with ServingHead(directory, device="cpu", calibration=calibration) as head:
        assert head.max_options == 255
        actual = torch.tensor(head.score("choice", query, candidates, baseline=baseline))
    expected = reference_logits(state, query, candidates, baseline)
    temperature = (1.25 if count <= 26 else 1.75) if calibration == "calibrated" else 1
    torch.testing.assert_close(actual, torch.stack((expected, expected.softmax(0),
                                                   (expected / temperature).softmax(0))), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("strength", [-0.5, 0.0, 0.25, 1.0, 1.5])
def test_v2_image_residual_and_calibration(v2_bundle, strength):
    directory, metadata, state = v2_bundle
    metadata["residual_strength"]["image_text"] = strength
    (directory / "config.json").write_text(json.dumps(metadata))
    query, candidates, baseline = features(metadata["hidden_size"], 5)
    with ServingHead(directory, device="cpu", calibration="calibrated") as head:
        actual = torch.tensor(head.score("choice", query, candidates, baseline=baseline, modality="image_text"))
    old = reference_logits(state, query, candidates, baseline)
    expected = baseline + strength * (old - baseline)
    torch.testing.assert_close(actual, torch.stack((expected, expected.softmax(0), (expected / 2).softmax(0))),
                               rtol=2e-6, atol=2e-6)
    if strength == 0:
        assert torch.equal(actual[0], baseline)


def test_v2_scores_do_not_depend_on_option_order(v2_bundle):
    directory, metadata, _ = v2_bundle
    query, candidates, baseline = features(metadata["hidden_size"], 255)
    baseline.zero_()
    order = torch.cat((torch.arange(128, 255), torch.arange(0, 64), torch.arange(64, 128)))
    with ServingHead(directory, device="cpu", calibration="calibrated") as head:
        expected = torch.tensor(head.score("choice", query, candidates, baseline=baseline))
        actual = torch.tensor(head.score("choice", query, candidates[order], baseline=baseline[order]))
    torch.testing.assert_close(actual[:, order.argsort()], expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual[1:].sum(-1), torch.ones(2), rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("calibration", ["raw", "calibrated"])
@pytest.mark.parametrize("empty", [False, True])
def test_v2_calibration_ranges(v2_bundle, calibration, empty):
    directory, metadata, state = v2_bundle
    metadata["capacities"]["ordinal"]["image_text"] = {"implemented_max": 5, "validated_max": None}
    (directory / "config.json").write_text(json.dumps(metadata))
    strata = [dict(primitive="choice", modality=modality, k_min=low, k_max=high, temperature=t)
              for modality, low, high, t in (("text", 2, 2, 1.), ("text", 4, 30, 2.),
                                            ("image_text", 2, 2, 3.))]
    (directory / "calibration.json").write_text(json.dumps({"strata": [] if empty else strata}))
    cases = (("choice", "text", 2, 1.), ("choice", "text", 3, None),
             ("choice", "text", 4, 2.), ("choice", "text", 30, 2.),
             ("choice", "text", 31, None),
             ("choice", "image_text", 2, 3.), ("noul", "text", 2, None),
             ("ordinal", "image_text", 5, None))
    with ServingHead(directory, device="cpu", calibration=calibration) as head:
        for primitive, modality, count, fitted in cases:
            fitted = None if empty or calibration == "raw" else fitted
            assert head.temperature_for(primitive, modality=modality, count=count) == fitted
            query, candidates, baseline = features(metadata["hidden_size"], count)
            if count > 26:
                baseline.zero_()
            expected = reference_logits(state, query, candidates, baseline)
            result = head.score(primitive, query, candidates, baseline=baseline, modality=modality)
            temperature = fitted if fitted is not None else 1.
            torch.testing.assert_close(torch.tensor(result), torch.stack((
                expected, expected.softmax(0), (expected / temperature).softmax(0))), rtol=2e-6, atol=2e-6)
            if fitted is None:
                assert result[1] == result[2]


def test_v2_overlapping_calibration_cannot_choose_arbitrary_temperature(v2_bundle):
    directory = v2_bundle[0]
    path = directory / "calibration.json"
    calibration = json.loads(path.read_text())
    calibration["strata"].append({**calibration["strata"][0], "temperature": 9.0})
    path.write_text(json.dumps(calibration))
    with pytest.raises(ValueError, match="Overlapping calibration"):
        ServingHead(directory, device="cpu", calibration="calibrated")


def test_v1_bundle_with_list_aware_encoding_is_rejected(bundle):
    directory, metadata, _ = bundle
    metadata["encoding_version"] = LIST_AWARE_VERSION
    (directory / "config.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="requires a v2 head bundle"):
        ServingHead(directory, device="cpu")
