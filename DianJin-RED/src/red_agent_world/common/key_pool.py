"""Round-robin API key pool used by local rollout runners."""

import asyncio
import logging
from typing import List


logger = logging.getLogger(__name__)


class KeyPool:
    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("api key list cannot be empty")
        self.keys = keys
        self.index = 0
        self.lock = asyncio.Lock()
        logger.info("KeyPool initialized with %d API keys", len(keys))

    async def get_key(self) -> str:
        async with self.lock:
            key = self.keys[self.index]
            self.index = (self.index + 1) % len(self.keys)
            return key
