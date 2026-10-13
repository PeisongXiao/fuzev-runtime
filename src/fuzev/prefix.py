"""Retained token inputs and image embeddings for decision requests."""

import secrets
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from .api import APIError


DEFAULT_PREFIX_CAPACITY = 128


@dataclass(frozen=True)
class Prefix:
    id: str
    token_ids: tuple[int, ...]
    payload: Any = None


class PrefixRegistry:
    def __init__(self, *, capacity=DEFAULT_PREFIX_CAPACITY):
        self.capacity = capacity
        self.entries = OrderedDict()
        self.lock = threading.RLock()

    def retain(self, token_ids, payload=None):
        tokens = tuple(token_ids)
        with self.lock:
            prefix = Prefix("prefix_" + secrets.token_hex(16), tokens, payload)
            self.entries[prefix.id] = prefix
            while len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
            return prefix

    def get(self, key):
        with self.lock:
            try:
                prefix = self.entries[key]
            except KeyError:
                raise APIError(404, {"type": "invalid_request_error", "message": "Unknown or evicted prefix"}) from None
            self.entries.move_to_end(key)
            return prefix

    def release(self, key):
        with self.lock:
            self.get(key)
            del self.entries[key]
            return True

    def clear(self):
        with self.lock:
            self.entries.clear()

