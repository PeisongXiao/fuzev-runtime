# FuZev Runtime

FuZev Runtime provides the request types, SystemOne adapter and decision head
shared by both [FuZev](https://github.com/PeisongXiao/fuzev) integrations. The
[vLLM](https://github.com/PeisongXiao/fuzev-vllm-plugin) and
[ExLlamaV3](https://github.com/PeisongXiao/fuzev-exllamav3-plugin) integrations
connect them to a running model.

- The [published head](https://huggingface.co/peisongxiao/Qwen3.8-Flash-Next-FuZev)
  targets Qwen3.8-Flash-Next, with hidden size 2,560.
- Heads for other backbones come from the
  [training guide](https://github.com/PeisongXiao/fuzev-research-archive/blob/main/docs/train-a-shared-head.md)
  and, in v1.0, load through the runtime Python API.

**Contents**

- [Install](#install)
- [SystemOne Mapping](#systemone-mapping)
- [Scoring From Python](#scoring-from-python)
- [Tests](#tests)
- [Limitations](#limitations)
- [License](#license)

## Install

Use Python 3.11 or newer (tested with 3.12). The head runs on CPU or CUDA;
engine integrations need CUDA builds.

Follow the canonical
[installation](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md) and
[server startup](https://github.com/PeisongXiao/fuzev/blob/main/docs/install.md#start-a-server)
steps for the runtime and your engine integration. The package installs the
`fuzev` Python namespace.

The [API guide](https://github.com/PeisongXiao/fuzev/blob/main/docs/api.md) covers
TypeSafe SDK (`typesafe-sdk`) and curl examples, HTTP request and response
fields, errors and extensions.

## SystemOne Mapping

`fuzev.systemone.SystemOneRequest` parses a request. `SystemOneAdapter` maps its
questions to the integration's asynchronous decision function. The integration
supplies the configured model name, head settings and tokenizer-based token
counter.

For an adapter configured by your integration:

```python
from fuzev.systemone import SystemOneAdapter, SystemOneRequest


async def answer_questions(
    adapter: SystemOneAdapter, payload: dict
) -> dict:
    request = SystemOneRequest.model_validate(payload)
    return await adapter.answer(request)
```

The adapter:

- Renders state and criteria in [Kev](https://github.com/jaredpalmer/kev)'s text
  format, with empty instructions when omitted.
- Orders Noul options `[Yes, No]`; the first probability is Yes.
- Decides questions concurrently and independently; each sees only the
  request's state.
- Returns answers in request order.
- Keeps structured Score legend values as objects and arrays.

The application owns the engine and service lifecycle. Reuse the configured
adapter across requests, and close the integration and head at shutdown.

## Scoring From Python

`fuzev.head.ServingHead` scores features supplied by an engine integration. Load
the head once when configuring the integration and reuse it across requests.

| Input | Shape |
| --- | --- |
| `query_features` | `[hidden_size]` |
| `option_features` | `[K, hidden_size]` |
| `lm_label_logits` | `[K]`, aligned with the options |

- The feature width matches the head bundle.
- K runs from 2 to the bundle's `max_options` for Choice and Score; Noul
  requires exactly 2. Callers must supply a valid K.
- `device` defaults to CUDA when available, otherwise CPU.
- The scorer detaches the features and language model scores and reads them on
  the head's device as FP32.

```python
from fuzev.head import ServingHead

# At application startup:
head = ServingHead("/path/to/fuzev-head", device="cpu", calibration="raw")

def score_choice(
    head: ServingHead, query_features, option_features, lm_label_logits
):
    logits, raw_probs, probs = head.score(
        "choice", query_features, option_features, baseline=lm_label_logits
    )
    return logits, raw_probs, probs

# At application shutdown:
head.close()
```

`score` returns three lists in option order: logits, raw softmax probabilities,
and probabilities under the selected calibration mode.

- `calibration="raw"`: `probs` equals `raw_probs`.
- `calibration="calibrated"`: applies the head bundle's temperatures.

The same scorer supports `"noul"` and `"ordinal"` (ordered Score).

The integration extracts query and candidate features from the model's final
hidden state and supplies the aligned language model scores. Each decision runs
one pass over the query prompt and one pass per option that extends it.

## Tests

Run the behavior suite from this source directory:

```bash
python -m pip install '.[test]'
python -m pytest
```

All tests pass. The six published-head golden cases skip unless
`FUZEV_HEAD_DIRECTORY` points to the downloaded head bundle.

## Limitations

- FuZev v1.0 is an experimental research release.
- v1.0 accepts 2 to 16 Choice options or Score levels per question. The head
  bundle declares the maximum (`max_options` in its `config.json`); the
  published head declares 16. Larger counts are planned for v1.1. Hosted Jev
  and Kev allow different counts; see the
  [capacity comparison](https://github.com/PeisongXiao/fuzev/blob/main/docs/api.md#capacity-comparison).
- Noul has exactly two outcomes.
- Tested on NVIDIA GB10 (Python 3.12, PyTorch 2.13.0+cu130).
- A head must be trained for the selected backbone.
- [Encodings](https://github.com/PeisongXiao/fuzev/blob/main/docs/models.md#encodings)
  describes runtime scoring and engine-serving formats.
  [API limitations](https://github.com/PeisongXiao/fuzev/blob/main/docs/api.md#limitations)
  covers images and retained prefixes.

Research methods are in the [archive](https://github.com/PeisongXiao/fuzev-research-archive).

## License

Code and the published head use [Apache-2.0](LICENSE). Backbone and dataset
licenses apply separately; see [model information](https://github.com/PeisongXiao/fuzev/blob/main/docs/models.md).
The bundled Qwen chat template is by froggeric and contributors under
Apache-2.0; see [NOTICE](NOTICE).
