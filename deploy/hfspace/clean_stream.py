import re

_GARBAGE_RE = re.compile(
    r"[\u0080-\u024F\u0370-\u1FFF\u2E80-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF\uFE00-\uFEFF"
    r"\uFF00-\uFFEF\u0E00-\u0E7F\u0980-\u09FF\u0600-\u06FF]"
)

# Repeat catches small runs (2-16 chars) repeating 3+ times, e.g. "user user user".
_REPEAT_RE = re.compile(r"(.{2,16})\1{2,}")

# Longer repeated phrases/sentences (20-200 chars) repeating 3+ times.
_LONG_REPEAT_RE = re.compile(r"(.{20,200}\.?)\1{2,}")

# The model re-entering self-chat and asking itself a new question.
_SELFCHAT_RE = re.compile(r"\nQuestion:\s")

# Known degenerate/injected markers the model emits when it enters self-chat:
#   "(iParam"  appears between the answer and the looping tail.
#   "<LM"      also seen as a stray start-of-turn artifact.
_BAD_MARKERS = ("(iParam", "<LM")


def _is_garbage(ch: str) -> bool:
    return _GARBAGE_RE.match(ch) is not None


def _degenerate_at(text: str) -> int:
    garbage_run = 0
    for i, ch in enumerate(text):
        if _is_garbage(ch):
            garbage_run += 1
            if garbage_run >= 2:
                return i - garbage_run + 1
        else:
            garbage_run = 0
    for marker in _BAD_MARKERS:
        at = text.find(marker)
        if at != -1:
            return at
    at = _SELFCHAT_RE.search(text)
    if at:
        return at.start()
    repeat_seen = _REPEAT_RE.search(text) or _LONG_REPEAT_RE.search(text)
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