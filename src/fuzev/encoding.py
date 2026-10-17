"""Qwen ChatML decision encodings.

`fuzev-candidate-branches-v1` labels options A-Z (up to 26).
`fuzev-list-aware-v2` uses the same A-Z prompt up to 26 options and a
numbered-list prompt with a zero baseline above 26.
"""
from pathlib import Path


VERSION = "fuzev-candidate-branches-v1"
LIST_AWARE_VERSION = "fuzev-list-aware-v2"
LABEL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LABEL_CAPACITY = len(LABEL_ALPHABET)
TEMPLATE = Path(__file__).resolve().parent / "templates/chat_template.jinja"
TEMPLATE_KWARGS = {"enable_thinking": False, "reasoning_effort": "none", "preserve_thinking": True}
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def labels(count):
    if not 1 <= count <= LABEL_CAPACITY:
        raise ValueError(f"Option labels support 1 to {LABEL_CAPACITY} options")
    return tuple(LABEL_ALPHABET[:count])


def query_suffix(question, candidates, additional_context=""):
    choices = "\n".join(f"({label}) {candidate['text']}" for label, candidate in zip(labels(len(candidates)), candidates))
    evidence = f"Additional evidence:\n{additional_context}\n" if additional_context else ""
    return ("\n<|im_start|>user\nSelect one candidate. Return only its label.\n"
            f"{evidence}Question: {question}\nCandidates:\n{choices}"
            "<|im_end|>\n<|im_start|>assistant\nAnswer:")


def candidate_suffix(candidate):
    return "\nCandidate under consideration:\n" + candidate["text"]


def list_aware_query_suffix(question, candidates, additional_context=""):
    choices = "\n".join(f"{index + 1}. {candidate['text']}" for index, candidate in enumerate(candidates))
    evidence = f"Additional evidence:\n{additional_context}\n" if additional_context else ""
    return ("\n<|im_start|>user\nSelect one candidate.\n"
            f"{evidence}Question: {question}\nCandidates:\n{choices}"
            "<|im_end|>\n<|im_start|>assistant\nAnswer:")


