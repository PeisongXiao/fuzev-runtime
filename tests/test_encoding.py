from fuzev.encoding import candidate_suffix, query_suffix


def test_candidate_branch_encoding_golden():
    candidates = [{"id": "a", "text": "Red"}, {"id": "b", "text": "Blue"}]
    assert query_suffix("Which color?", candidates) == (
        "\n<|im_start|>user\nSelect one candidate. Return only its label.\n"
        "Question: Which color?\nCandidates:\n(A) Red\n(B) Blue"
        "<|im_end|>\n<|im_start|>assistant\nAnswer:")
    assert query_suffix("Which color?", candidates, "The sky is clear.") == (
        "\n<|im_start|>user\nSelect one candidate. Return only its label.\n"
        "Additional evidence:\nThe sky is clear.\n"
        "Question: Which color?\nCandidates:\n(A) Red\n(B) Blue"
        "<|im_end|>\n<|im_start|>assistant\nAnswer:")
    assert [candidate_suffix(candidate) for candidate in candidates] == [
        "\nCandidate under consideration:\nRed",
        "\nCandidate under consideration:\nBlue",
    ]
