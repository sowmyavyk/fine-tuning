import re

_GARBAGE_RE = re.compile(
    r"[\u0080-\u024F\u0370-\u1FFF\u2E80-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF\uFE00-\uFEFF"
    r"\uFF00-\uFFEF\u0E00-\u0E7F\u0980-\u09FF\u0600-\u06FF]"
)

_REPEAT_RE = re.compile(r"(.{2,16})\1{2,}")


def _is_garbage(ch: str) -> bool:
    return _GARBAGE_RE.match(ch) is not None


def _degenerate_at(text: str) -> int:
    garbage_run = 0
    repeat_seen = _REPEAT_RE.search(text)
    for i, ch in enumerate(text):
        if _is_garbage(ch):
            garbage_run += 1
            if garbage_run >= 3:
                return i - garbage_run + 1
        else:
            garbage_run = 0
    if repeat_seen:
        return repeat_seen.start()
    return -1


def clean_text(text: str) -> str:
    at = _degenerate_at(text)
    return text[:at].strip() if at != -1 else text.strip()


def clean_stream(stream, window: int = 64):
    buffer = []
    for piece in stream:
        buffer.append(piece)
        if len(buffer) >= window:
            chunk = "".join(buffer)
            at = _degenerate_at(chunk)
            if at != -1:
                yield chunk[:at]
                return
    tail = "".join(buffer)
    at = _degenerate_at(tail)
    yield tail[:at] if at != -1 else tail
