import re

# Source text is 370 words; ±10% = 333–407
SOURCE_WORD_COUNT = 370
MIN_WORDS = 333
MAX_WORDS = 407

# Common preamble/wrapper patterns to strip before length-counting
_PREAMBLE_PATTERNS = [
    # Markdown code fences
    re.compile(r"^```[^\n]*\n(.*?)\n?```\s*$", re.DOTALL),
    # Surrounding double quotes
    re.compile(r'^"(.*)"$', re.DOTALL),
    # Surrounding single quotes
    re.compile(r"^'(.*)'$", re.DOTALL),
    # "Here is the paraphrase:" and variants
    re.compile(
        r"^(?:here(?:'s| is) (?:the )?paraphrase[:\s]*|"
        r"paraphrase[:\s]+|"
        r"here(?:'s| is) (?:a )?(?:revised |revised version[:\s]*|version[:\s]*)?(?:of the )?(?:paraphrase[:\s]*|text[:\s]*)|"
        r"revised paraphrase[:\s]*|"
        r"sure[,!]?\s+here(?:'s| is)[^:]*[:\s]*)",
        re.IGNORECASE,
    ),
]


# Qwen3 think-block pattern: <think>...</think> (may span multiple lines)
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def strip_wrappers(text: str) -> tuple[str, bool]:
    """Strip preamble/wrapper artifacts. Returns (stripped_text, was_stripped)."""
    stripped = text.strip()
    original = stripped

    # Strip <think>...</think> blocks (Qwen3 thinking-mode tokens)
    stripped = _THINK_BLOCK.sub("", stripped).strip()

    # Try fence (highest priority after think-strip)
    m = _PREAMBLE_PATTERNS[0].match(stripped)
    if m:
        stripped = m.group(1).strip()

    # Surrounding quotes
    for pat in _PREAMBLE_PATTERNS[1:3]:
        m = pat.match(stripped)
        if m:
            stripped = m.group(1).strip()
            break

    # Preamble line (strip only the first line if it matches)
    lines = stripped.split("\n")
    m = _PREAMBLE_PATTERNS[3].match(lines[0])
    if m:
        stripped = "\n".join(lines[1:]).strip()

    return stripped, stripped != original


def count_words(text: str) -> int:
    """Whitespace-split word count — simple and reproducible."""
    return len(text.split())


def ends_with_sentence(text: str) -> bool:
    """Return True if text ends with terminal punctuation."""
    stripped = text.rstrip()
    return bool(stripped) and stripped[-1] in ".!?"


def truncate_to_range(text: str, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> str:
    """Hard-truncate to nearest sentence boundary inside [min_words, max_words]."""
    # Split into sentences (keep delimiters)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    result_sentences: list[str] = []
    running_count = 0

    for sent in sentences:
        wc = count_words(sent)
        if running_count + wc > max_words:
            break
        result_sentences.append(sent)
        running_count += wc

    truncated = " ".join(result_sentences)

    # If we still haven't hit min_words, something is very wrong — return as-is
    if count_words(truncated) < min_words:
        return truncated

    return truncated
