import json
import math
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch.nn import functional as F

from fuzev._scoring import SharedDecisionHead
from fuzev.head import ServingHead


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


# Reference outputs of the published head on CPU for the inputs from features().
GOLDEN = {
    "choice": [
        [-1.2030912638, -0.5956468582, 0.0001845047, 0.5985535383],
        [0.0817905590, 0.1501457095, 0.2724452317, 0.4956185818],
        [0.0997843966, 0.1667452008, 0.2759186029, 0.4575517476],
    ],
    "noul": [
        [-1.2009224892, 0.6009224653],
        [0.1416266263, 0.8583733439],
        [0.1955216080, 0.8044783473],
    ],
    "ordinal": [
        [-1.2032563686, -1.0735722780, -0.9555011392, -0.8348923922,
         -0.7130817175, -0.6012115479, -0.4771792591, -0.3566844761,
         -0.2497879118, -0.1256862581, 0.0000024263, 0.1211143658,
         0.2377401441, 0.3589374125, 0.4703834057, 0.6026751995],
        [0.0218548756, 0.0248810910, 0.0279992968, 0.0315883383,
         0.0356802978, 0.0399036929, 0.0451730713, 0.0509577058,
         0.0567067116, 0.0641994178, 0.0727975965, 0.0821703747,
         0.0923347548, 0.1042318568, 0.1165201068, 0.1330008656],
        [0.0207824241, 0.0237896461, 0.0269044470, 0.0305076409,
         0.0346367508, 0.0389194638, 0.0442894995, 0.0502150320,
         0.0561322570, 0.0638818964, 0.0728217959, 0.0826178119,
         0.0932944119, 0.1058538258, 0.1188897043, 0.1364634186],
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
