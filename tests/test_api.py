import asyncio
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from fuzev.api import APIError, api_error, models_response, request_id, validate_decision, validation_error
from fuzev.contracts import DecisionRequest
from fuzev.service_contracts import GenerationRequest


def decision(count=2, **changes):
    value = {
        "model": "actual-model", "primitive": "choice",
        "messages": [{"role": "user", "content": "Pick an option"}],
        "candidates": [{"id": str(index), "text": str(index)} for index in range(count)],
    }
    value.update(changes)
    return DecisionRequest.model_validate(value)


@pytest.fixture
def head():
    metadata = {"max_options": 17, "release_date": "2030-04-05"}
    return SimpleNamespace(head_id="reference", primitives=["choice", "noul", "ordinal"],
                           max_options=metadata["max_options"], metadata=metadata)


@pytest.mark.parametrize("error,status,detail", [
    (APIError(409, "Request already active"), 409, "Request already active"),
    (ExceptionGroup("Questions failed", [APIError(503, "Engine unavailable")]), 503, "Engine unavailable"),
    (asyncio.CancelledError(), 499, "Request cancelled"),
    (ValueError("Private diagnostic"), 500, "Internal server error"),
])
def test_shared_error_responses(error, status, detail):
    assert api_error(error) == (status, {"detail": detail})


def test_validation_errors_name_the_field():
    assert api_error(validation_error("state", "Input exceeds the context window")) == (
        422, {"detail": [{"loc": ["body", "state"],
                          "msg": "Input exceeds the context window", "type": "value_error"}]},
    )
    with pytest.raises(ValidationError) as error:
        decision(1)
    status, body = api_error(error.value)
    assert status == 422
    assert body["detail"][0]["loc"] == ["body", "candidates"]
    malformed_json = json.JSONDecodeError("Expected value", "{", 1)
    status, body = api_error(malformed_json)
    assert status == 422
    assert body["detail"][0]["loc"] == ["body"]


def test_decision_option_ceiling_uses_bundle_metadata(head):
    max_options = head.max_options
    validate_decision(decision(max_options), head)
    with pytest.raises(APIError) as error:
        validate_decision(decision(max_options + 1), head)
    status, body = api_error(error.value)
    assert status == 422
    assert body["detail"][0]["loc"] == ["body", "candidates"]
    assert body["detail"][0]["msg"] == f"This head supports at most {max_options} options"


def test_decision_accepts_an_optional_client_alias(head):
    for model in (None, "client-alias"):
        validate_decision(decision(model=model), head)


def test_request_id_header_is_optional_and_preserved():
    assert request_id({"X-TypeSafe-Request-ID": "job.A:1_b-c"}) == "job.A:1_b-c"
    generated = request_id({})
    assert isinstance(generated, str) and generated
    with pytest.raises(APIError) as error:
        request_id({"X-TypeSafe-Request-ID": "has spaces"})
    status, body = api_error(error.value)
    assert status == 422
    assert body["detail"][0]["loc"] == ["header", "x-typesafe-request-id"]


def test_models_response_supports_openai_and_sdk_discovery(head):
    result = models_response("actual-model", head)
    assert result["object"] == "list"
    assert result["data"][0]["id"] == "actual-model"
    assert result["data"][0]["object"] == "model"
    assert result["models"] == [{"name": "actual-model",
                                "description": "FuZev decisions with actual-model",
                                "release_date": "2030-04-05"}]
    assert models_response("actual-model")["models"] == []


def test_openai_generation_fields_normalize_to_engine_inputs():
    request = GenerationRequest.model_validate({
        "messages": [{"role": "developer", "content": [{"type": "text", "text": "Be "},
                       {"type": "text", "text": "brief."}], "name": "editor"}],
        "max_completion_tokens": 12, "stop": "END", "n": 1,
        "logprobs": False, "stream_options": None, "user": "client", "presence_penalty": 0,
        "tools": [], "tool_choice": "none", "response_format": {"type": "text"},
    })
    assert request.messages[0].model_dump() == {"role": "system", "content": "Be brief."}
    assert request.max_tokens == 12 and request.stop == ["END"]
    assert request.model is None
    assert GenerationRequest(prompt="Hi", stop=None).stop == []


def test_duplicate_candidate_identifiers():
    with pytest.raises(ValidationError, match="identifiers must be unique"):
        decision(candidates=[{"id": "a", "text": "A"}, {"id": "a", "text": "B"}])


def test_noul_yes_no_order():
    with pytest.raises(ValidationError, match="Noul candidates"):
        decision(primitive="noul", candidates=[{"id": "no", "text": "No"}, {"id": "yes", "text": "Yes"}])


def test_decision_context_is_messages_or_prefix():
    for changes in ({"messages": None}, {"prefix_id": "also-messages"}):
        with pytest.raises(ValidationError, match="exactly one of messages or prefix_id"):
            decision(**changes)


def test_generation_uses_prompt_or_messages():
    for value in ({}, {"prompt": "Hi", "messages": [{"role": "user", "content": "Hi"}]}):
        with pytest.raises(ValidationError, match="exactly one of prompt or text messages"):
            GenerationRequest.model_validate(value)


@pytest.mark.parametrize("field", [{"temperature": 2.5}, {"temperature": -0.1}, {"top_k": -5}])
def test_out_of_range_sampling_values_are_rejected(field):
    with pytest.raises(ValidationError):
        GenerationRequest.model_validate({"prompt": "Hi", **field})


def test_chat_image_url_part_becomes_the_request_image():
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "auto"}}
    request = GenerationRequest.model_validate({"messages": [
        {"role": "user", "content": [{"type": "text", "text": "Describe "}, image, {"type": "text", "text": "this."}]}]})
    assert request.image == "data:image/png;base64,AAAA"
    assert request.messages[0].content == "Describe this."
    for extra in ({"image": "AAAA"}, {"messages": [{"role": "user", "content": [image, image]}]}):
        with pytest.raises(ValidationError, match="Send one image"):
            GenerationRequest.model_validate({"messages": [{"role": "user", "content": [image]}], **extra})


def test_decision_head_id_is_accepted_and_ignored(head):
    validate_decision(decision(head_id="client-label"), head)


def test_head_declares_supported_primitives(head):
    head.primitives = ("choice",)
    with pytest.raises(APIError) as error:
        validate_decision(decision(primitive="ordinal"), head)
    assert error.value.status == 422
    assert error.value.detail[0]["loc"] == ["body", "primitive"]
