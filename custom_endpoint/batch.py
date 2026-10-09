"""Batch alignment and concurrent execution.

A call node declares ``INPUT_IS_LIST = True``, so ComfyUI hands it the raw
lists of every input.  This module aligns them (length 1 broadcasts, length N
must match) and runs one worker thread per item, keeping the output order equal
to the input order.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Sequence

MAX_PARALLEL_WORKERS = 16


def as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def broadcast(value: Any, length: int) -> list:
    """Broadcast a list to ``length`` items (1 repeats, mismatch raises)."""
    items = as_list(value)
    if length <= 0:
        return []
    if len(items) == length:
        return items
    if len(items) == 1:
        return items * length
    raise ValueError(
        f"Batch inputs must all have the same length or length 1; got {len(items)} "
        f"against a batch of {length}."
    )


def resolve_batch(values: dict[str, Any]) -> tuple[int, dict[str, list]]:
    """Return ``(batch_length, aligned_values)`` for a kwargs mapping."""
    lengths = [len(as_list(value)) for value in values.values() if isinstance(value, list)]
    length = max(lengths) if lengths else 0
    if length == 0:
        # No list input at all (every input missing) - single empty job.
        length = 1
    aligned = {name: broadcast(value, length) for name, value in values.items()}
    return length, aligned


def run_concurrent(
    worker: Callable[[Any], Any],
    items: Sequence[Any],
    max_workers: int = 4,
) -> list[Any]:
    """Run ``worker`` on every item concurrently, results in input order.

    A worker that raises still yields a result slot: the exception object is
    stored so the caller can turn it into a log line.
    """
    count = len(items)
    if count == 0:
        return []
    if count == 1 or max_workers <= 1:
        return [_call(worker, items[0])]

    workers = max(1, min(int(max_workers), MAX_PARALLEL_WORKERS, count))
    results: list[Any] = [None] * count
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_call, worker, item): index for index, item in enumerate(items)}
        for future, index in futures.items():
            try:
                results[index] = future.result()
            except Exception as error:  # pragma: no cover - _call swallows
                results[index] = error
    return results


def _call(worker: Callable[[Any], Any], item: Any) -> Any:
    try:
        return worker(item)
    except Exception as error:  # noqa: BLE001 - failures are per-item data
        return error


def split_batch(items: Iterable[Any]) -> list[Any]:
    return list(items)
