"""Shared generation and decision request contracts."""

import logging
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator

from .contracts import DecisionRequest


class ServiceDecisionRequest(DecisionRequest):
    image: str | None = None


class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    url: str


class ImagePart(BaseModel):
    """OpenAI image content part; the URL is a data URL or bare base64."""
    type: Literal["image_url"]
    image_url: ImageURL


class TextMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    role: Literal["system", "developer", "user", "assistant"]
    content: str | list[TextPart | ImagePart]
    _images: list[str] = PrivateAttr(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def extra_fields(cls, data):
        if isinstance(data, dict) and (ignored := set(data) - set(cls.model_fields)):
            logging.getLogger(__name__).info("Ignoring message fields: %s", ", ".join(sorted(ignored)))
        return data

    @model_validator(mode="after")
    def normalize(self):
        if self.role == "developer":
            self.role = "system"
        if isinstance(self.content, list):
            self._images = [part.image_url.url for part in self.content if isinstance(part, ImagePart)]
            self.content = "".join(part.text for part in self.content if isinstance(part, TextPart))
        return self


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    model: str | None = Field(default=None, min_length=1)
    prompt: str | None = Field(default=None, min_length=1)
    messages: list[TextMessage] | None = Field(default=None, min_length=1)
    image: str | None = None
    max_tokens: int | None = Field(default=None, ge=1, validation_alias=AliasChoices("max_completion_tokens", "max_tokens"))
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    seed: int | None = None
    stop: list[str] = Field(default_factory=list, max_length=4)
    chat_template_kwargs: dict = Field(default_factory=dict)
    retain_decision_prefix: bool = False
    stream: bool = False

    @model_validator(mode="before")
    @classmethod
    def optional_fields(cls, data):
        if isinstance(data, dict):
            for key in ("tools", "tool_choice", "response_format", "logprobs", "top_logprobs"):
                if data.get(key) not in (None, False, [], "none", {"type": "text"}):
                    raise ValueError(f"{key} is not supported by this generation endpoint")
            if data.get("n") not in (None, 1):
                raise ValueError("This generation endpoint supports n=1")
            ignored = set(data) - set(cls.model_fields) - {"max_completion_tokens", "n", "tools", "tool_choice", "response_format", "logprobs", "top_logprobs"}
            if ignored:
                logging.getLogger(__name__).info("Ignoring generation fields: %s", ", ".join(sorted(ignored)))
        return data

    @field_validator("stop", mode="before")
    @classmethod
    def normalize_stop(cls, value):
        return [] if value is None else [value] if isinstance(value, str) else value

    @field_validator("chat_template_kwargs")
    @classmethod
    def template_arguments(cls, value):
        reserved = {"chat_template", "tokenize", "return_tensors", "return_dict"}.intersection(value)
        if reserved:
            raise ValueError("Reserved chat template arguments: " + ", ".join(sorted(reserved)))
        return value

    @model_validator(mode="after")
    def exactly_one_context(self):
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("Provide exactly one of prompt or text messages")
        images = [url for message in self.messages or () for url in message._images]
        if images:
            if len(images) > 1 or self.image is not None:
                raise ValueError("Send one image, as `image` or one `image_url` content part")
            self.image = images[0]
        return self


