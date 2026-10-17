import asyncio
import math

import pytest
from pydantic import ValidationError

from fuzev.api import api_error
from fuzev.systemone import SystemOneAdapter, SystemOneRequest


class Decisions:
    def __init__(self, probabilities=None):
        self.calls = []
        self.probabilities = probabilities or {}

    async def __call__(self, request, request_id):
        self.calls.append((request, request_id))
        values = self.probabilities.get(
            request.primitive, [1 / len(request.candidates)] * len(request.candidates)
        )
        selected = max(range(len(values)), key=values.__getitem__)
        return {
            "id": request_id,
            "model": request.model,
            "primitive": request.primitive,
            "head_id": request.head_id,
            "calibration_id": "calibrated",
            "selected_id": request.candidates[selected].id,
            "scores": [
                {
                    "id": candidate.id,
                    "logit": math.log(value),
                    "raw_probability": value,
                    "calibrated_probability": value,
                }
                for candidate, value in zip(request.candidates, values)
            ],
            "execution": {"submitted_tokens": 17, "input_tokens": 9},
        }


def request(**changes):
    value = {
        "state": {"subject": "Charged twice", "urgent": True, "amounts": [12, 12]},
        "questions": {
            "team": {
                "type": "choice",
                "criteria": {
                    "billing": {"department": "Accounts", "tags": ["invoice", True]},
                    "shipping": None,
                },
            },
            "urgent": {
                "type": "noul",
                "criteria": {"true": "Urgent", "false": {"priority": "Later"}},
            },
            "rating": {
                "type": "score",
                "criteria": [
                    "low",
                    {"label": "medium", "checks": [True, None]},
                    ["high", {"value": 3}],
                ],
            },
        },
    }
    value.update(changes)
    return SystemOneRequest.model_validate(value)


def test_kev_encoding_and_golden_answers():
    backend = Decisions({"choice": [0.2, 0.8], "noul": [0.9, 0.1], "ordinal": [0.1, 0.2, 0.7]})
    adapter = SystemOneAdapter(backend, "custom-finetune", lambda text: 23)
    result = asyncio.run(adapter.answer(request()))
    calls = {call.primitive: call for call, _ in backend.calls}
    assert all(call.question == "" for call in calls.values())
    assert all(
        call.messages == [{"role": "user", "content": (
            "subject: Charged twice\nurgent: True\namounts:\n  - 12\n  - 12"
        )}]
        for call in calls.values()
    )
    # Nested JSON uses the Kev state and option text format.
    assert [(c.id, c.text) for c in calls["choice"].candidates] == [
        ("billing", "billing: department: Accounts\ntags:\n  - invoice\n  - True"),
        ("shipping", "shipping"),
    ]
    assert [(c.id, c.text) for c in calls["noul"].candidates] == [
        ("yes", "Yes: Urgent"), ("no", "No: priority: Later"),
    ]
    assert [(c.id, c.text) for c in calls["ordinal"].candidates] == [
        ("0", "low"),
        ("1", "label: medium\nchecks:\n  - True\n  - "),
        ("2", "- high\n- value: 3"),
    ]
    assert result["model"] == "custom-finetune"
    assert list(result["answers"]) == ["team", "urgent", "rating"]
    assert result["answers"] == {
        "team": {
            "type": "choice", "choice": "shipping", "confidence": pytest.approx(0.6),
            "probabilities": {"billing": 0.2, "shipping": 0.8},
        },
        "urgent": {"type": "noul", "noul": 0.9},
        "rating": {
            "type": "score", "score": pytest.approx(1.6), "confidence": pytest.approx(0.4),
            "legend": {"0": "low", "1": {"label": "medium", "checks": [True, None]},
                       "2": ["high", {"value": 3}]},
            "probabilities": {"0": 0.1, "1": 0.2, "2": 0.7},
        },
    }
    assert result["usage"] == {"input_tokens": 27, "output_tokens": 23}
    assert result["latency_ms"] >= 0


def test_aliases_and_unknown_fields_with_retained_prefix():
    req = request(
        model="client-defined-alias", metadata={"ignored": True}, prefix_id="retained-prefix",
        state="Additional evidence",
        questions={"q": {"type": "noul", "instructions": {"question": "Is it urgent?"},
                         "future_setting": True}},
    )
    backend = Decisions()
    result = asyncio.run(
        SystemOneAdapter(backend, "actual-model", len).answer(req)
    )
    call, _ = backend.calls[0]
    assert result["model"] == call.model == "actual-model"
    assert result["answers"] == {"q": {"type": "noul", "noul": 0.5}}
    assert call.question == "question: Is it urgent?"
    assert [(c.id, c.text) for c in call.candidates] == [("yes", "Yes"), ("no", "No")]
    assert call.prefix_id == "retained-prefix"
    assert call.messages is None
    assert call.additional_context == "Additional evidence"


def test_score_legend_scalar_values_use_kev_text():
    req = request(questions={"rating": {"type": "score", "criteria": [None, True, 3, 2.5]}})
    result = asyncio.run(SystemOneAdapter(Decisions(), "model", len).answer(req))
    assert result["answers"]["rating"] == {
        "type": "score", "score": 1.5, "confidence": 0.0,
        "legend": {"0": "", "1": "True", "2": "3", "3": "2.5"},
        "probabilities": {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25},
    }


@pytest.mark.parametrize("kind,criteria", [
    ("choice", {"only": None}), ("score", ["only"]), ("score", "wrong type"),
])
def test_known_criteria_remain_validated(kind, criteria):
    with pytest.raises(ValidationError) as error:
        request(questions={"q": {"type": kind, "criteria": criteria}})
    status, body = api_error(error.value)
    assert status == 422
    assert body["detail"][0]["loc"][-1] == "criteria"


def test_independent_questions_are_submitted_concurrently():
    async def exercise():
        backend = Decisions()
        all_started = asyncio.Event()
        started = set()

        async def decide(decision, identity):
            started.add(decision.primitive)
            if len(started) == 3:
                all_started.set()
            await all_started.wait()
            return await backend(decision, identity)

        adapter = SystemOneAdapter(decide, "actual-model", len)
        result = await asyncio.wait_for(adapter.answer(request()), timeout=2)
        assert list(result["answers"]) == ["team", "urgent", "rating"]

    asyncio.run(exercise())


def test_question_failure_cancels_siblings():
    async def exercise():
        all_started, blocked = asyncio.Event(), asyncio.Event()
        started, cancelled = set(), set()

        async def decide(decision, identity):
            started.add(decision.primitive)
            if len(started) == 3:
                all_started.set()
            try:
                await all_started.wait()
                if decision.primitive == "noul":
                    raise RuntimeError("Question failed")
                await blocked.wait()
            except asyncio.CancelledError:
                cancelled.add(decision.primitive)
                raise

        adapter = SystemOneAdapter(decide, "actual-model", len)
        with pytest.raises(ExceptionGroup) as error:
            await asyncio.wait_for(adapter.answer(request()), timeout=2)
        assert cancelled == {"choice", "ordinal"}
        assert api_error(error.value) == (500, {"detail": "Internal server error"})

    asyncio.run(exercise())
