from fuzev.encoding import candidate_suffix, list_aware_query_suffix, query_suffix


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


def test_list_aware_27_encoding_golden():
    candidates = [{"id": str(index), "text": f"Option {index:02d}"} for index in range(1, 28)]
    assert list_aware_query_suffix("Which option?", candidates, "Shared evidence.") == (
        "\n<|im_start|>user\nSelect one candidate.\n"
        "Additional evidence:\nShared evidence.\n"
        "Question: Which option?\nCandidates:\n"
        "1. Option 01\n2. Option 02\n3. Option 03\n4. Option 04\n5. Option 05\n"
        "6. Option 06\n7. Option 07\n8. Option 08\n9. Option 09\n10. Option 10\n"
        "11. Option 11\n12. Option 12\n13. Option 13\n14. Option 14\n15. Option 15\n"
        "16. Option 16\n17. Option 17\n18. Option 18\n19. Option 19\n20. Option 20\n"
        "21. Option 21\n22. Option 22\n23. Option 23\n24. Option 24\n25. Option 25\n"
        "26. Option 26\n27. Option 27"
        "<|im_end|>\n<|im_start|>assistant\nAnswer:")
    assert list_aware_query_suffix("Which option?", candidates) == list_aware_query_suffix(
        "Which option?", candidates, "Shared evidence.").replace("Additional evidence:\nShared evidence.\n", "")
    assert candidate_suffix(candidates[-1]) == "\nCandidate under consideration:\nOption 27"
