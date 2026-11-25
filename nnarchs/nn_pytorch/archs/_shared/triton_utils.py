try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None
    TRITON_AVAILABLE = False
else:
    TRITON_AVAILABLE = True


def from_triton(error: BaseException) -> bool:
    """Whether the error, or one it was raised from, comes from Triton itself: a
    Triton kernel that fails to compile or launch switches itself off, while other
    errors (out of memory, wrong inputs) are raised."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__module__.partition(".")[0] == "triton":
            return True
        current = current.__cause__ or current.__context__
    return False
