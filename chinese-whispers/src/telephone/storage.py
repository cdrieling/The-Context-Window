import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parents[2] / "results" / "runs"


def run_dir(timestamp: str, config_name: str, seed: int) -> Path:
    name = f"{timestamp}_{config_name}_seed{seed}"
    d = RESULTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_round(
    directory: Path,
    round_num: int,
    config_name: str,
    model_config_name: str,
    api_model_id: str,
    seed: int,
    input_text: str,
    output_text: str,
    retries: int,
    truncated: bool,
    inference_seconds: float,
    tokens_per_second: float | None,
    temperature: float,
    tps_estimated: bool = False,
) -> dict:
    from telephone.length_control import count_words

    record = {
        "round": round_num,
        "config_name": config_name,
        "model_config_name": model_config_name,
        "api_model_id": api_model_id,
        "seed": seed,
        "input_text": input_text,
        "input_word_count": count_words(input_text),
        "output_text": output_text,
        "output_word_count": count_words(output_text),
        "retries": retries,
        "truncated": truncated,
        "inference_seconds": round(inference_seconds, 3),
        "tokens_per_second": round(tokens_per_second, 1) if tokens_per_second is not None else None,
        "tps_estimated": tps_estimated,
        "temperature": temperature,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    path = directory / f"round_{round_num:02d}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    log.info("Wrote %s", path)
    return record


def write_summary(
    directory: Path,
    config_name: str,
    seed: int,
    source_text: str,
    rounds: list[dict],
    started: str,
    finished: str,
) -> Path:
    summary = {
        "config_name": config_name,
        "seed": seed,
        "source_text": source_text,
        "started": started,
        "finished": finished,
        "round_count": len(rounds),
        "any_truncated": any(r["truncated"] for r in rounds),
        "rounds": rounds,
    }
    path = directory / "run_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info("Summary written to %s", path)
    return path
