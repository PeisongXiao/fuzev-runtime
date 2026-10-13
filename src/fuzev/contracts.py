from enum import StrEnum
from typing import Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_HEAD_ID = "reference"


class Primitive(StrEnum):
    CHOICE = "choice"
    NOUL = "noul"
    ORDINAL = "ordinal"


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    text: str


class DecisionMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str | None = Field(default=None, min_length=1)
    primitive: Primitive
    question: str = ""
    candidates: list[Candidate] = Field(min_length=2)
    prefix_id: str | None = None
    messages: list[DecisionMessage] | None = Field(default=None, min_length=1)
    additional_context: str = ""
    head_id: str = DEFAULT_HEAD_ID

    @model_validator(mode="after")
    def validate_semantics(self):
        ids = [candidate.id for candidate in self.candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("Candidate identifiers must be unique")
        if self.primitive == Primitive.NOUL:
            if ids != ["yes", "no"]:
                raise ValueError("Noul candidates must be [yes, no] in that order")
        if (self.prefix_id is None) == (self.messages is None):
            raise ValueError("Provide exactly one of messages or prefix_id")
        return self


