# FuZev Runtime

FuZev Runtime provides the request types, SystemOne adapter and decision head
for [FuZev](https://github.com/PeisongXiao/fuzev/tree/v1.1). The
[ExLlamaV3](https://github.com/PeisongXiao/fuzev-exllamav3-plugin/tree/v1.1)
integration connects them to a running model.

- The [published head](https://huggingface.co/peisongxiao/Qwen3.8-Flash-Next-FuZev/tree/v1.1)
  targets Qwen3.8-Flash-Next, with hidden size 2,560.
- Heads for other backbones come from the
  [training guide](https://github.com/PeisongXiao/fuzev-research-archive/blob/v1.1/docs/train-a-shared-head.md)
  and, in v1.1, load through the runtime Python API.

**Contents**

- [Install](#install)
- [SystemOne Mapping](#systemone-mapping)
- [Scoring From Python](#scoring-from-python)
- [Tests](#tests)
- [Limitations](#limitations)
- [What's New In 1.1](#whats-new-in-11)
- [License](#license)

## Install

Use Python 3.11 or newer (tested with 3.12). The head runs on CPU or CUDA;
the ExLlamaV3 integration needs CUDA builds.

Follow the canonical
[installation](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/install.md) and
[server startup](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/install.md#start-a-server)
steps for the runtime and the ExLlamaV3 integration. The package installs the
`fuzev` Python namespace.

The [API guide](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/api.md) covers
TypeSafe SDK (`typesafe-sdk`) and curl examples, HTTP request and response
fields, errors and extensions.

## SystemOne Mapping

`fuzev.systemone.SystemOneRequest` parses a request. `SystemOneAdapter` maps its
questions to the integration's asynchronous decision function. The integration
supplies the configured model name, head ID and tokenizer-based token counter.

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
- Reports `usage.input_tokens` as the sum of each decision's
  `execution.input_tokens`; the ExLlamaV3 integration counts the shared state
  once.

The application owns the engine and service lifecycle. Reuse the configured
adapter across requests, and close the integration and head at shutdown.

## Scoring From Python

`fuzev.head.ServingHead` scores features supplied by an engine integration. Load
the head once when configuring the integration and reuse it across requests.

| Input | Shape |
| --- | --- |
| `query_features` | `[hidden_size]` |
| `option_features` | `[K, hidden_size]` |
| `lm_label_logits` | `[K]`, aligned with the options; zeros above 26 options |

- The feature width matches the head bundle.
- K runs from 2 to the bundle's capacity for Choice and Score; Noul
  requires exactly 2. Callers must supply a valid K.
- `modality` is `text` by default; pass the experimental `image_text` for
  image-bearing requests, including retained image prefixes.
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
- `calibration="calibrated"`: applies the head bundle's temperatures; requests
  outside the bundle's calibrated ranges use raw probabilities.

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
`FUZEV_HEAD_DIRECTORY` points to the downloaded head bundle:

```bash
hf download peisongxiao/Qwen3.8-Flash-Next-FuZev --revision v1.1 --local-dir fuzev-head
FUZEV_HEAD_DIRECTORY="$PWD/fuzev-head" python -m pytest tests/test_head.py -k golden
```

## Limitations

- FuZev v1.1 is an experimental research release.
- v1.1 accepts 2 to 255 Choice options or Score levels per question. The head
  bundle declares the maximum (`implemented_max` per question type and modality
  in its `config.json`; see
  [Decision Head](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/models.md#decision-head));
  the published head declares 255 for text and 26 with an image. Hosted Jev
  and Kev allow different counts; see the
  [capacity comparison](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/api.md#capacity-comparison).
- Noul has exactly two outcomes.
- Images (`image_text`) are experimental; run at your own risk.
- Tested on NVIDIA GB10 (Python 3.12, PyTorch 2.13.0+cu130).
- A head must be trained for the selected backbone.
- [Encodings](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/models.md#encodings)
  describes runtime scoring and engine-serving formats.
  [API limitations](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/api.md#limitations)
  covers images and retained prefixes.

Research methods are in the [archive](https://github.com/PeisongXiao/fuzev-research-archive/tree/v1.1).

## What's New In 1.1

- `SystemOneAdapter(decide, model_name, token_count, head_id)` takes no
  `max_options` or `primitives`; the integration admits each question with
  `fuzev.api.validate_decision` against the loaded bundle.
  fuzev-exllamav3-plugin 1.0 and fuzev-vllm-plugin 1.0 work with
  fuzev-runtime 1.0 only
  ([upgrading from 1.0](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/releases.md#upgrading-from-10)).
- `ServingHead` loads `fuzev-shared-head-package-v2` bundles, which declare
  `implemented_max` per question type and modality
  ([decision head](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/models.md#decision-head)).
  Schema-v1 bundles with one `max_options` still load; head 1.0.0 admits up
  to 16 options.
- Head 1.1.0 uses the `fuzev-list-aware-v2` encoding and scores up to 255 text
  options: A-Z labels with label logits up to 26 options, and a numbered list
  (`fuzev.encoding.list_aware_query_suffix`) with a zero baseline above 26
  ([encodings](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/models.md#encodings)).
- `ServingHead.score` takes `modality`: `text` by default, or the experimental
  `image_text` ([Scoring From Python](#scoring-from-python)).
- Calibrated mode applies the bundle's temperature for the question type,
  modality and option-count range; requests outside those ranges use raw
  probabilities ([Scoring From Python](#scoring-from-python)).
- `ServingHead` reads `head_id` from the bundle's `config.json`
  (`fuzev-head-1.1.0` for head 1.1.0, `reference` when absent) and takes no
  `head_id` argument
  ([lower-level decisions](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/api.md#lower-level-decisions)).
- `usage.input_tokens`, the encoded request length, sums each decision's
  `execution.input_tokens`; `execution.submitted_tokens` keeps the logical
  branch count
  ([usage](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/api.md#usage)).
- The six golden cases in `tests/test_head.py` hold head 1.1.0 outputs;
  [Tests](#tests) downloads it with `--revision v1.1`.

## License

Code and the published head use [Apache-2.0](LICENSE). Backbone and dataset
licenses apply separately; see [model information](https://github.com/PeisongXiao/fuzev/blob/v1.1/docs/models.md).
The bundled Qwen chat template is by froggeric and contributors under
Apache-2.0; see [NOTICE](NOTICE).
