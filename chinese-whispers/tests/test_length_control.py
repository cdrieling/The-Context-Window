from telephone.length_control import (
    count_words,
    ends_with_sentence,
    strip_wrappers,
    truncate_to_range,
    MIN_WORDS,
    MAX_WORDS,
)


def test_count_words_basic():
    assert count_words("hello world") == 2
    assert count_words("  spaced   words  ") == 2
    assert count_words("") == 0


def test_count_words_hyphenated():
    # Hyphenated terms count as one word (documented behavior)
    assert count_words("gold-plated copper disc") == 3


def test_ends_with_sentence():
    assert ends_with_sentence("This is a sentence.")
    assert ends_with_sentence("Really?")
    assert ends_with_sentence("Wow!")
    assert not ends_with_sentence("no terminal punctuation")
    assert not ends_with_sentence("")
    assert not ends_with_sentence("   ")


def test_strip_wrappers_no_wrapper():
    text = "Plain text without wrapper."
    cleaned, was_stripped = strip_wrappers(text)
    assert cleaned == text
    assert not was_stripped


def test_strip_wrappers_markdown_fence():
    text = "```\nThis is the paraphrase.\n```"
    cleaned, was_stripped = strip_wrappers(text)
    assert cleaned == "This is the paraphrase."
    assert was_stripped


def test_strip_wrappers_double_quotes():
    text = '"This is quoted."'
    cleaned, was_stripped = strip_wrappers(text)
    assert cleaned == "This is quoted."
    assert was_stripped


def test_strip_wrappers_preamble():
    text = "Here is the paraphrase:\nVoyager 1 launched in 1977."
    cleaned, was_stripped = strip_wrappers(text)
    assert "Here is the paraphrase" not in cleaned
    assert was_stripped


def test_strip_wrappers_preamble_variant():
    text = "Here's the paraphrase:\nVoyager 1 launched in 1977."
    cleaned, was_stripped = strip_wrappers(text)
    assert "Here's the paraphrase" not in cleaned
    assert was_stripped


def test_truncate_to_range_already_in_range():
    # Build a text that's exactly 370 words in range
    words = ["word"] * 370
    text = " ".join(words[:MIN_WORDS]) + "."
    result = truncate_to_range(text)
    assert MIN_WORDS <= count_words(result) <= MAX_WORDS


def test_truncate_to_range_too_long():
    # Build a long text with sentence boundaries (10 words × 50 = 500 words > MAX_WORDS)
    sentences = ["This is a sentence with ten words in it here."] * 50
    text = " ".join(sentences)
    assert count_words(text) > MAX_WORDS
    result = truncate_to_range(text)
    assert count_words(result) <= MAX_WORDS


def test_strip_wrappers_think_block():
    text = "<think>\nAnalyzing the request...\n</think>\nVoyager 1 launched in 1977."
    cleaned, was_stripped = strip_wrappers(text)
    assert "<think>" not in cleaned
    assert "Voyager 1 launched" in cleaned
    assert was_stripped


def test_range_constants():
    assert MIN_WORDS == 333
    assert MAX_WORDS == 407
