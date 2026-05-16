import os
import time
import logging

from openai import OpenAI, APIConnectionError, APIStatusError

log = logging.getLogger(__name__)

_MAX_CONNECTION_RETRIES = 2
_RETRY_BACKOFF = 2.0  # seconds


def make_client() -> OpenAI:
    url = os.environ.get("OMLX_API_URL")
    key = os.environ.get("OMLX_API_KEY")
    if not url:
        raise SystemExit("OMLX_API_URL is not set. Add it to .env.")
    if not key:
        raise SystemExit("OMLX_API_KEY is not set. Add it to .env.")
    return OpenAI(base_url=url, api_key=key, timeout=120.0)


def probe_models(client: OpenAI) -> list[str]:
    """Return list of available model IDs from the server."""
    try:
        models = client.models.list()
        return [m.id for m in models.data]
    except APIConnectionError as exc:
        url = os.environ.get("OMLX_API_URL", "unknown")
        raise SystemExit(
            f"oMLX server unreachable at {url}. Start the server or check OMLX_API_URL."
        ) from exc


def complete(
    client: OpenAI,
    model_id: str,
    prompt: str,
    seed: int,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> tuple[str, float, float | None]:
    """
    Returns (output_text, inference_seconds, tokens_per_second | None).
    tokens_per_second is None if usage stats are unavailable.
    """
    attempt = 0
    while True:
        try:
            start = time.monotonic()
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
            )
            elapsed = time.monotonic() - start

            text = response.choices[0].message.content or ""

            tps: float | None = None
            if response.usage and response.usage.completion_tokens:
                tps = response.usage.completion_tokens / elapsed
            else:
                # Estimate: ~1.3 tokens per word (rough BPE estimate)
                estimated_tokens = len(text.split()) * 1.3
                tps = estimated_tokens / elapsed
                log.debug("tokens_per_second estimated (no usage stats from server)")

            return text, elapsed, tps

        except APIConnectionError as exc:
            attempt += 1
            if attempt > _MAX_CONNECTION_RETRIES:
                url = os.environ.get("OMLX_API_URL", "unknown")
                raise SystemExit(
                    f"oMLX server unreachable at {url} after {attempt} attempts."
                ) from exc
            log.warning("Connection error (attempt %d), retrying in %.0fs…", attempt, _RETRY_BACKOFF)
            time.sleep(_RETRY_BACKOFF)

        except APIStatusError as exc:
            raise SystemExit(f"API error {exc.status_code}: {exc.message}") from exc
