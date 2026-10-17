import pytest

from fuzev.prefix import PrefixRegistry
from fuzev.api import APIError, api_error


def test_single_prefix_capacity_retains_the_latest_input():
    registry = PrefixRegistry(capacity=1)
    first = registry.retain([1])
    assert registry.get(first.id).token_ids == (1,)
    second = registry.retain([2, 3])
    with pytest.raises(APIError) as error:
        registry.get(first.id)
    assert error.value.status == 404
    assert registry.get(second.id).token_ids == (2, 3)
    assert registry.release(second.id) is True


def test_retained_inputs_lru_and_release():
    registry = PrefixRegistry(capacity=2)
    tokens, payload = [1, 2, 3], object()
    first = registry.retain(tokens, payload=payload)
    tokens.append(4)
    assert first.token_ids == (1, 2, 3)
    assert first.payload is payload

    second = registry.retain([5])
    assert registry.get(first.id) is first
    third = registry.retain([6])
    with pytest.raises(APIError) as error:
        registry.get(second.id)
    assert api_error(error.value) == (404, {"error": {"type": "invalid_request_error", "message": "Unknown or evicted prefix"}})
    assert registry.get(third.id) is third

    assert registry.release(first.id) is True
    with pytest.raises(APIError):
        registry.release(first.id)
    with pytest.raises(APIError):
        registry.get(first.id)
    assert registry.get(third.id) is third

    last = registry.retain([7])
    registry.clear()
    with pytest.raises(APIError):
        registry.get(last.id)
