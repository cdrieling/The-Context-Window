PARAPHRASE_TEMPLATE = """\
You will paraphrase the following text. Follow these rules exactly:

1. Output ONLY the paraphrased text. No preamble, no explanation, no notes.
2. Target length: {target_words} words. Acceptable range: {min_words} to {max_words} words.
3. Preserve every fact, name, date, and number exactly. Do not round numbers.
4. Use different wording and sentence structures than the original.
5. Do NOT add any information that is not in the source text.

TEXT:
{text}

PARAPHRASE:"""

RETRY_SUFFIX = "\n\nYour previous output was {actual} words. The required range is {min_words}–{max_words} words."


def build_paraphrase_prompt(
    text: str,
    target_words: int = 370,
    min_words: int = 333,
    max_words: int = 407,
) -> str:
    return PARAPHRASE_TEMPLATE.format(
        target_words=target_words,
        min_words=min_words,
        max_words=max_words,
        text=text,
    )


def build_retry_prompt(
    text: str,
    actual_words: int,
    target_words: int = 370,
    min_words: int = 333,
    max_words: int = 407,
) -> str:
    base = PARAPHRASE_TEMPLATE.format(
        target_words=target_words,
        min_words=min_words,
        max_words=max_words,
        text=text,
    )
    suffix = RETRY_SUFFIX.format(
        actual=actual_words,
        min_words=min_words,
        max_words=max_words,
    )
    return base + suffix
