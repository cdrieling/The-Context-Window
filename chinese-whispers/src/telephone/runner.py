import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from telephone.inference import complete, make_client, probe_models
from telephone.length_control import (
    MIN_WORDS,
    MAX_WORDS,
    SOURCE_WORD_COUNT,
    count_words,
    ends_with_sentence,
    strip_wrappers,
    truncate_to_range,
)
from telephone.prompts import build_paraphrase_prompt, build_retry_prompt
from telephone.storage import run_dir, write_round, write_summary

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parents[2]
_CONFIG_DIR = _PROJECT_ROOT / "config"
_DATA_DIR = _PROJECT_ROOT / "data"

_MAX_RETRIES = 3
_TEMPERATURE = 0.7
_MAX_TOKENS = 1024


def load_config():
    with open(_CONFIG_DIR / "models.yaml") as f:
        models = yaml.safe_load(f)["models"]
    with open(_CONFIG_DIR / "experiments.yaml") as f:
        experiments = yaml.safe_load(f)["experiments"]
    return models, experiments


def load_source_text() -> str:
    return (_DATA_DIR / "source_text.txt").read_text().strip()


def _in_range(wc: int) -> bool:
    return MIN_WORDS <= wc <= MAX_WORDS


def run_single(
    experiment_name: str,
    seed: int,
    rounds: int = 10,
    dry_run: bool = False,
) -> Path:
    models_cfg, experiments_cfg = load_config()
    if experiment_name not in experiments_cfg:
        raise SystemExit(f"Unknown experiment '{experiment_name}'. Available: {list(experiments_cfg)}")

    exp = experiments_cfg[experiment_name]
    chain = exp["chain"]

    if rounds > len(chain):
        raise SystemExit(f"--rounds {rounds} exceeds chain length {len(chain)}")

    source_text = load_source_text()
    client = make_client()

    # Probe + validate
    available = probe_models(client)
    log.info("Available models: %s", available)

    # Collect model IDs needed for this run
    needed_ids = {models_cfg[m]["api_model_id"] for m in chain[:rounds]}
    for model_id in needed_ids:
        if model_id not in available:
            raise SystemExit(
                f"Model '{model_id}' not loaded on server.\nAvailable: {available}"
            )

    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    directory = run_dir(timestamp, experiment_name, seed)
    log.info("Run directory: %s", directory)

    current_text = source_text
    round_records: list[dict] = []

    for round_num in range(1, rounds + 1):
        model_key = chain[round_num - 1]
        model_cfg = models_cfg[model_key]
        api_model_id = model_cfg["api_model_id"]

        log.info("--- Round %d/%d | model=%s | seed=%d ---", round_num, rounds, api_model_id, seed)

        prompt = build_paraphrase_prompt(current_text)
        output_text = ""
        inference_seconds = 0.0
        tokens_per_second = None
        retries = 0
        truncated = False
        stripped_logged = False

        for attempt in range(_MAX_RETRIES + 1):
            raw, elapsed, tps = complete(
                client, api_model_id, prompt, seed=seed,
                temperature=_TEMPERATURE, max_tokens=_MAX_TOKENS,
            )
            inference_seconds = elapsed
            tokens_per_second = tps

            # Strip wrappers
            cleaned, was_stripped = strip_wrappers(raw)
            if was_stripped and not stripped_logged:
                log.info("Round %d: stripped preamble/wrapper from output", round_num)
                stripped_logged = True

            wc = count_words(cleaned)
            in_range = _in_range(wc)
            complete_sentence = ends_with_sentence(cleaned)

            if in_range and complete_sentence:
                output_text = cleaned
                retries = attempt
                break

            if attempt < _MAX_RETRIES:
                reason = "length" if not in_range else "incomplete sentence"
                log.warning(
                    "Round %d attempt %d: %s (%d words, ends_with_sentence=%s) — retrying",
                    round_num, attempt + 1, reason, wc, complete_sentence,
                )
                prompt = build_retry_prompt(current_text, actual_words=wc)
                retries = attempt + 1
            else:
                # Hard truncate
                log.warning(
                    "Round %d: max retries reached (%d words), hard-truncating", round_num, wc
                )
                output_text = truncate_to_range(cleaned)
                truncated = True
                retries = _MAX_RETRIES

        record = write_round(
            directory=directory,
            round_num=round_num,
            config_name=experiment_name,
            model_config_name=model_key,
            api_model_id=api_model_id,
            seed=seed,
            input_text=current_text,
            output_text=output_text,
            retries=retries,
            truncated=truncated,
            inference_seconds=inference_seconds,
            tokens_per_second=tokens_per_second,
            temperature=_TEMPERATURE,
        )
        round_records.append(record)

        wc_out = count_words(output_text)
        log.info(
            "Round %d done: %d words | %.1fs | %.0f t/s%s%s",
            round_num,
            wc_out,
            inference_seconds,
            tokens_per_second or 0,
            " [TRUNCATED]" if truncated else "",
            " [retries=%d]" % retries if retries else "",
        )

        current_text = output_text

    finished = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary_path = write_summary(
        directory=directory,
        config_name=experiment_name,
        seed=seed,
        source_text=source_text,
        rounds=round_records,
        started=started,
        finished=finished,
    )

    log.info("Run complete. Summary: %s", summary_path)
    return summary_path
