"""Small async retry helper."""

import asyncio
import logging
from functools import wraps
from typing import Any, Callable


logger = logging.getLogger(__name__)


def async_retry(
    max_retries: int = 3,
    delay: float = 2.0,
    exceptions: tuple = (Exception,),
    backoff: float = 1.0,
):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt >= max_retries:
                        logger.error("%s failed after %d retries: %s", func.__name__, max_retries, exc)
                        break
                    logger.warning(
                        "%s failed, retry %d/%d: %s",
                        func.__name__,
                        attempt + 1,
                        max_retries,
                        exc,
                    )
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff
            raise last_exception

        return wrapper

    return decorator
