import hashlib
import random


def shuffle_with_seed(lst: list, *seed_parts: str) -> list:
    """Return a shuffled copy of lst using a deterministic, process-stable seed.

    Python's built-in hash() for strings is randomised per process (PYTHONHASHSEED),
    which means two server runs produce different shuffle orders for the same input.
    hashlib.sha256 is stable across restarts, so the same email+exam_id always
    produces the same ordering.
    """
    combined = "".join(seed_parts)
    seed = int(hashlib.sha256(combined.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    result = list(lst)
    rng.shuffle(result)
    return result
