"""Qwen ChatML candidate-branches-v1 encoding used to train the released head.

Options use labels A-Z. Heads with more than 26 options need a new encoding version.
"""
from pathlib import Path


VERSION = "fuzev-candidate-branches-v1"
TEMPLATE = Path(__file__).resolve().parent / "templates/chat_template.jinja"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def labels(count):
    if not 1 <= count <= 26:
        raise ValueError("Candidate-branches-v1 supports 1 to 26 option labels")
    return tuple(chr(ord("A") + index) for index in range(count))


def query_suffix(question, candidates, additional_context=""):
    choices = "\n".join(f"({label}) {candidate['text']}" for label, candidate in zip(labels(len(candidates)), candidates))
    evidence = f"Additional evidence:\n{additional_context}\n" if additional_context else ""
    return ("\n<|im_start|>user\nSelect one candidate. Return only its label.\n"
            f"{evidence}Question: {question}\nCandidates:\n{choices}"
            "<|im_end|>\n<|im_start|>assistant\nAnswer:")


def candidate_suffix(candidate):
    return "\nCandidate under consideration:\n" + candidate["text"]


