"""Jev-compatible questions and answers using the evaluated decision encoding."""

import asyncio
import copy
import json
import time
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from .api import APIError, validation_error
from .contracts import Candidate, DEFAULT_HEAD_ID, Primitive
from .service_contracts import ServiceDecisionRequest


def state_text(value, indent=0):
    """Render JSON with the Kev-style encoding used to evaluate the head."""
    pad = "  " * indent
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(f"{pad}- {state_text(item, indent + 1).lstrip()}" for item in value)
    return "\n".join(
        f"{pad}{key}:\n{state_text(item, indent + 1)}"
        if isinstance(item, (dict, list)) else f"{pad}{key}: {state_text(item)}"
        for key, item in value.items()
    )


def _option_text(key, description):
    return key if description is None or description == "" else f"{key}: {state_text(description)}"


class QuestionBase(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    instructions: JsonValue = None


class Choice(QuestionBase):
    type: Literal["choice"]
    criteria: dict[str, JsonValue] = Field(min_length=2)


class Noul(QuestionBase):
    type: Literal["noul"]
    criteria: dict[str, JsonValue] | None = None


class OrderedScore(QuestionBase):
    type: Literal["score"]
    criteria: list[JsonValue] = Field(min_length=2)


Question = Annotated[Choice | Noul | OrderedScore, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    state: JsonValue
    model: str | None = Field(default=None, min_length=1)
    questions: dict[str, Question] = Field(min_length=1)
    prefix_id: str | None = None
    image: str | None = None


def decision_request(request, question, model_name, head_id, state):
    criteria = question.criteria
    if question.type == "noul":
        criteria = criteria or {}
        options = [{"id": "yes", "text": _option_text("Yes", criteria.get("true"))},
                   {"id": "no", "text": _option_text("No", criteria.get("false"))}]
    elif question.type == "choice":
        options = [{"id": key, "text": _option_text(key, description)}
                   for key, description in criteria.items()]
    else:
        options = [{"id": str(index), "text": state_text(level)}
                   for index, level in enumerate(criteria)]
    return ServiceDecisionRequest.model_construct(
        model=model_name, primitive=Primitive("ordinal" if question.type == "score" else question.type),
        head_id=head_id, question=state_text(question.instructions),
        candidates=[Candidate.model_construct(**option) for option in options],
        prefix_id=request.prefix_id, image=request.image,
        messages=None if request.prefix_id is not None else [{"role": "user", "content": state}],
        additional_context=state if request.prefix_id is not None else "",
    )


def answer(question, values, selected):
    if question.type == "noul":
        return {"type": "noul", "noul": values[0]}
    count = len(values)
    if question.type == "choice":
        keys = list(question.criteria)
        confidence = (values[selected] - 1 / count) / (1 - 1 / count)
        return {"type": "choice", "choice": keys[selected],
                "confidence": max(0.0, min(1.0, confidence)),
                "probabilities": dict(zip(keys, values))}
    uniform_deviation = sum(abs(index - (count - 1) / 2) for index in range(count)) / count
    confidence = max(0.0, 1 - sum(p * abs(index - selected) for index, p in enumerate(values))
                     / uniform_deviation)
    return {"type": "score", "score": sum(index * p for index, p in enumerate(values)),
            "confidence": confidence,
            "legend": {str(index): copy.deepcopy(level) if isinstance(level, (str, dict, list))
                       else state_text(level) for index, level in enumerate(question.criteria)},
            "probabilities": {str(index): p for index, p in enumerate(values)}}


def input_tokens(response):
    return response["execution"]["submitted_tokens"]


class SystemOneAdapter:
    """Submit independent questions through the configured decision service."""

    def __init__(self, decide, model_name, token_count, head_id=DEFAULT_HEAD_ID, *, max_options,
                 primitives):
        self.decide, self.model_name, self.token_count = decide, model_name, token_count
        self.head_id, self.max_options = head_id, max_options
        self.primitives = primitives

    async def answer(self, request, *, prepare=None):
        started = time.perf_counter()
        for key, question in request.questions.items():
            primitive = "ordinal" if question.type == "score" else question.type
            if primitive not in self.primitives:
                raise validation_error(["questions", key, "type"], "This head does not support the requested primitive")
            count = 2 if question.type == "noul" else len(question.criteria)
            if count > self.max_options:
                raise validation_error(["questions", key, question.type, "criteria"],
                                       f"This head supports at most {self.max_options} options")
        state = state_text(request.state)
        prepared = [decision_request(request, question, self.model_name, self.head_id, state)
                    for question in request.questions.values()]
        submit = self.decide
        if prepare is not None:
            submit = await prepare(prepared)
        async def submit_question(key, decision):
            try:
                return await submit(decision, "decision_" + uuid4().hex)
            except APIError as error:
                if error.status == 422 and isinstance(error.detail, list):
                    for item in error.detail:
                        item["loc"] = ["body", "questions", key]
                raise
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(submit_question(key, decision))
                     for key, decision in zip(request.questions, prepared)]
        responses = [task.result() for task in tasks]
        answers = {}
        for (key, question), response in zip(request.questions.items(), responses):
            values = [row["calibrated_probability"] for row in response["scores"]]
            selected = max(range(len(values)), key=values.__getitem__)
            answers[key] = answer(question, values, selected)
        output_tokens = self.token_count(json.dumps(answers, ensure_ascii=False, separators=(",", ":")))
        return {"model": self.model_name, "answers": answers,
                "usage": {"input_tokens": sum(map(input_tokens, responses)), "output_tokens": output_tokens},
                "latency_ms": (time.perf_counter() - started) * 1000}
