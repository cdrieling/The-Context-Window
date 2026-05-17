from telephone.prompts import build_paraphrase_prompt, build_retry_prompt


def test_prompt_contains_text():
    prompt = build_paraphrase_prompt("The quick brown fox.")
    assert "The quick brown fox." in prompt


def test_prompt_contains_target_words():
    prompt = build_paraphrase_prompt("text", target_words=370, min_words=333, max_words=407)
    assert "370" in prompt
    assert "333" in prompt
    assert "407" in prompt


def test_prompt_template_immutable():
    # Run twice, confirm same output (no hidden state)
    p1 = build_paraphrase_prompt("same text")
    p2 = build_paraphrase_prompt("same text")
    assert p1 == p2


def test_retry_prompt_includes_actual_count():
    prompt = build_retry_prompt("some text", actual_words=250, min_words=333, max_words=407)
    assert "250" in prompt
    assert "333" in prompt
    assert "407" in prompt


def test_prompt_ends_with_paraphrase_marker():
    prompt = build_paraphrase_prompt("any text")
    assert prompt.strip().endswith("PARAPHRASE:")


def test_retry_prompt_still_ends_with_context():
    # Retry prompt appends the word-count note after the base prompt
    prompt = build_retry_prompt("some text", actual_words=200)
    assert "200 words" in prompt
    assert "333" in prompt
