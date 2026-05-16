#!/usr/bin/env python3
"""
grounding_gradient.py — MVP

Experiment: Zwei lokale LLMs führen einen Dialog. Ein Grounding-Agent extrahiert
numerische Behauptungen. Eskalationsstufen bestimmen, wie stark der Eingriff in
den weiterlaufenden Kontext ist.

MVP-Scope:
  - Zwei Modelle über OpenAI-kompatible Endpoints (Ollama / LM Studio / MLX)
  - Nur numerische Claim-Extraktion (Regex)
  - Zwei Eskalationsstufen: 0 (Baseline, nur Log) und 3 (Inline-Markierung)
  - Eine Metrik: Numeric Claim Density pro 100 Tokens
  - JSONL-Log pro Lauf

Run:
    export LMSTUDIO_API_KEY="dein-key"
    pip install openai
    python grounding_gradient.py --stufe 0
    python grounding_gradient.py --stufe 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

LM_STUDIO_ENDPOINT = "http://localhost:1234/v1"
LM_STUDIO_API_KEY  = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")


# model_id: exakt so, wie er in LM Studio gelistet ist (Rechtsklick → Copy identifier)
MODEL_A = {
    "name": "qwen3.6",
    "endpoint": LM_STUDIO_ENDPOINT,
    "model_id": "qwen3.6-35b-a3b-uncensored-hauhaucs-aggressive",       # ggf. anpassen
    "api_key": LM_STUDIO_API_KEY,
    "role": "Explorer",
}

MODEL_B = {
    "name": "gemma4",
    "endpoint": LM_STUDIO_ENDPOINT,
    "model_id": "google/gemma-4-26b-a4b",  # ggf. anpassen
    "api_key": LM_STUDIO_API_KEY,
    "role": "Skeptiker",
}

SYSTEM_PROMPT_TEMPLATE = """Du bist eine KI-Instanz im Dialog mit einer anderen \
KI-Instanz. Deine Rolle: {role}.

Rahmen:
- Ihr diskutiert ein offenes Thema mit Fokus auf analytische Tiefe.
- Als {role} agierst du entsprechend: Explorer entwirft Thesen und Methoden, \
Skeptiker prüft kritisch und fordert Präzision.
- Antworte mit deiner Nachricht an die andere Instanz. Wenn du intern denkst, \
verwende dafür <think>-Blöcke; diese werden gefiltert, bevor dein Gegenüber \
deine Antwort sieht.
- Halte dich kompakt: maximal ~300 Wörter im finalen Output pro Turn."""

STARTING_PROMPT = """Lasst uns folgende Frage zwischen uns ausarbeiten:
Wie könnte man die informationelle Dichte eines Dialogs zwischen zwei \
Sprachmodellen operationalisieren – also messbar machen? Entwickelt gemeinsam \
einen Ansatz, der auch dann funktioniert, wenn keine externe Referenz verfügbar ist."""

MAX_TURNS = 10
TEMPERATURE = 0.7
MAX_TOKENS = 3000          # lieber großzügig – abgeschnittene Turns killen den Lauf

# Kollaps-Detektion: wenn ein Turn leer ist oder ein einzelnes Token dominant
# wiederholt (>N-mal), brechen wir ab statt weiterzumachen.
MIN_TURN_CHARS       = 20
REPETITION_RUN_MIN   = 10   # N-mal dasselbe Token DIREKT hintereinander = Collapse

def detect_repetition(text: str) -> str | None:
    """Prüft auf degenerate Wiederholung. Signal: ein Token N-mal konsekutiv.
    Anders als reine Häufigkeits-Counts fängt das nur echte Repetition-Loops
    (Metrik-Metrik-Metrik-…) und nicht normale Artikel-Frequenz in deutscher Prosa."""
    if not text or len(text.strip()) < MIN_TURN_CHARS:
        return "empty_or_too_short"

    segments = [s for s in re.split(r"[-\s.,;:]+", text) if s]
    if len(segments) < REPETITION_RUN_MIN:
        return None

    current_tok = segments[0]
    current_run = 1
    for seg in segments[1:]:
        if seg == current_tok:
            current_run += 1
            if current_run >= REPETITION_RUN_MIN and len(current_tok) > 1:
                return f"repetition_run[{current_tok!r} x{current_run}+]"
        else:
            current_tok = seg
            current_run = 1
    return None

# Numerische Claims: Zahlenwerte mit optionalem Vergleichs- oder Einheitsmarker.
# Bewusst großzügig – lieber Rauschen in Kauf nehmen als echte Claims verpassen.
NUMERIC_PATTERN = re.compile(
    r"""
    (?<![A-Za-zÄÖÜäöü])                # nicht mitten in einem Wort
    (?:[≈~=]|<|>|ca\.|etwa|rund)?       # optionaler Marker davor
    \s*
    -?\d+(?:[.,]\d+)?                   # Zahl (dt. Komma oder Punkt)
    \s*
    (?:%|prozent|pct|σ|‰)?              # optionale Einheit
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Thinking-Blöcke: bekannte Varianten. Erweiterbar, wenn neue Modelle auftauchen.
THINKING_PATTERNS = [
    re.compile(r"<think>.*?</think>",       re.DOTALL | re.IGNORECASE),
    re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL | re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Datentypen
# ---------------------------------------------------------------------------

@dataclass
class NumericClaim:
    span: str
    position: int          # Character-Offset im clean_text

@dataclass
class TurnMetrics:
    token_estimate: int
    numeric_claim_count: int
    claim_density_per_100_tokens: float
    thinking_claim_count: int          # Claims im Thinking-Block (zum Vergleich)

@dataclass
class Turn:
    index: int
    speaker: str
    role: str
    raw_text: str          # vollständige Modellausgabe inkl. Thinking
    clean_text: str        # Thinking entfernt – Basis für Extraktion
    thinking_text: str     # extrahiertes Thinking, wird NICHT weitergereicht
    modified_text: str     # clean_text nach Eskalation – geht in nächsten Kontext
    claims: list[NumericClaim]
    metrics: TurnMetrics
    finish_reason: str     # "stop", "length", "content_filter", ...
    timestamp: str

# ---------------------------------------------------------------------------
# Claim-Extraktion, Thinking-Stripping & Metriken
# ---------------------------------------------------------------------------

def strip_thinking(text: str) -> tuple[str, str]:
    """Trennt Thinking-Blöcke vom Rest. Gibt (clean_text, thinking_joined) zurück."""
    thinking_parts: list[str] = []
    cleaned = text
    for pattern in THINKING_PATTERNS:
        thinking_parts.extend(m.group() for m in pattern.finditer(cleaned))
        cleaned = pattern.sub("", cleaned)
    return cleaned.strip(), "\n---\n".join(thinking_parts)

def extract_numeric_claims(text: str) -> list[NumericClaim]:
    claims: list[NumericClaim] = []
    for match in NUMERIC_PATTERN.finditer(text):
        span = match.group().strip()
        if not re.search(r"\d", span):
            continue
        claims.append(NumericClaim(span=span, position=match.start()))
    return claims

def estimate_tokens(text: str) -> int:
    # Grober Proxy: ~4 chars/token. Reicht für Density-Vergleiche zwischen Turns.
    return max(1, len(text) // 4)

def compute_metrics(clean: str, thinking: str, claims: list[NumericClaim]) -> TurnMetrics:
    tokens = estimate_tokens(clean)
    thinking_claims = len(extract_numeric_claims(thinking)) if thinking else 0
    return TurnMetrics(
        token_estimate=tokens,
        numeric_claim_count=len(claims),
        claim_density_per_100_tokens=round(len(claims) * 100 / tokens, 2),
        thinking_claim_count=thinking_claims,
    )

# ---------------------------------------------------------------------------
# Eskalations-Strategien
# ---------------------------------------------------------------------------

def escalation_stufe_0(text: str, claims: list[NumericClaim]) -> str:
    """Baseline: Text unverändert, nur Log."""
    return text

def escalation_stufe_3(text: str, claims: list[NumericClaim]) -> str:
    """Inline-Markierung: jeder numerische Claim erhält ein [UNVERIFIED]-Tag."""
    # Von hinten nach vorne einsetzen, damit Offsets stabil bleiben
    modified = text
    for claim in sorted(claims, key=lambda c: c.position, reverse=True):
        insert_at = claim.position + len(claim.span)
        modified = modified[:insert_at] + " [UNVERIFIED]" + modified[insert_at:]
    return modified

ESCALATION_STRATEGIES = {
    0: escalation_stufe_0,
    3: escalation_stufe_3,
}

def detect_repetition_legacy_wrapper(text: str) -> str | None:
    """Kompatibilitäts-Shim – wird über die neue Definition oben ersetzt."""
    return detect_repetition(text)

# ---------------------------------------------------------------------------
# LLM-Client
# ---------------------------------------------------------------------------

def call_model(model_cfg: dict, history: list[dict],
               debug: bool = False) -> tuple[str, str, str]:
    """Gibt (content, reasoning_content, finish_reason) zurück.

    reasoning_content wird von LM Studio bei Reasoning-Modellen als separates
    Feld geliefert – parallel zu etwaigen <think>-Tags im content."""
    client = OpenAI(base_url=model_cfg["endpoint"], api_key=model_cfg["api_key"])
    response = client.chat.completions.create(
        model=model_cfg["model_id"],
        messages=history,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    choice = response.choices[0]
    if debug:
        print(f"\n[DEBUG] Full choice object:\n{choice.model_dump_json(indent=2)}\n",
              file=sys.stderr)

    msg_dict = choice.message.model_dump()
    content   = (msg_dict.get("content") or "").strip()
    reasoning = (msg_dict.get("reasoning_content") or "").strip()
    finish    = choice.finish_reason or "unknown"
    return content, reasoning, finish

def call_model_robust(model_cfg: dict, history: list[dict],
                      debug: bool = False) -> tuple[str, str, str]:
    """Wrapper mit einem Retry bei Leerturn (empty+stop), weil LM Studio das
    reproduzierbar liefert, wenn Model-Switch oder KV-Cache quergeht."""
    content, reasoning, finish = call_model(model_cfg, history, debug=debug)
    if not content and finish == "stop":
        print(f"    ⟳ Leerturn bei finish=stop, Retry …", file=sys.stderr)
        time.sleep(0.5)
        content, reasoning, finish = call_model(model_cfg, history, debug=False)
        if content:
            finish = f"{finish}|retry_ok"
    return content, reasoning, finish

def build_history(system_prompt: str, turns: list[Turn], as_speaker: str) -> list[dict]:
    """OpenAI-Message-Liste aus Sicht von 'as_speaker'.
    Eigene Turns → role=assistant, fremde Turns → role=user."""
    history: list[dict] = [{"role": "system", "content": system_prompt}]
    for turn in turns:
        role = "assistant" if turn.speaker == as_speaker else "user"
        history.append({"role": role, "content": turn.modified_text})
    return history

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_experiment(stufe: int, run_dir: Path, debug_first_turn: bool = True) -> Path:
    apply_escalation = ESCALATION_STRATEGIES[stufe]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_stufe{stufe}"
    log_path = run_dir / f"{run_id}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    models = [MODEL_A, MODEL_B]
    turns: list[Turn] = []

    # Seed-Turn: Starting-Prompt als synthetischer Turn, damit Modell A reagieren kann
    seed = Turn(
        index=0,
        speaker="SEED",
        role="Setup",
        raw_text=STARTING_PROMPT,
        clean_text=STARTING_PROMPT,
        thinking_text="",
        modified_text=STARTING_PROMPT,
        claims=[],
        metrics=compute_metrics(STARTING_PROMPT, "", []),
        finish_reason="seed",
        timestamp=datetime.now().isoformat(),
    )
    turns.append(seed)

    print(f"[Setup] Stufe {stufe} | Log: {log_path}")
    print(f"[Seed]  {STARTING_PROMPT[:120]}...\n")

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(json.dumps(asdict(seed), ensure_ascii=False) + "\n")

        for turn_index in range(1, MAX_TURNS + 1):
            active_model = models[(turn_index - 1) % 2]
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(role=active_model["role"])
            history = build_history(system_prompt, turns, active_model["name"])

            t0 = time.time()
            try:
                raw, reasoning_api, finish = call_model_robust(
                    active_model, history,
                    debug=(debug_first_turn and turn_index == 1),
                )
            except Exception as exc:
                print(f"[Fehler] {active_model['name']}: {exc}", file=sys.stderr)
                break
            latency = time.time() - t0

            # Zwei Thinking-Quellen kombinieren:
            #   1. reasoning_content aus der API (LM Studio, echte Reasoning-Modelle)
            #   2. <think>-Tags im content (Fallback für Modelle, die inline wrappen)
            clean, thinking_inline = strip_thinking(raw)
            thinking = "\n---\n".join(filter(None, [reasoning_api, thinking_inline]))

            # Kollaps-Detektion: wenn wir hier nicht abbrechen, kippt das nächste
            # Modell reproduzierbar in Repetition.
            collapse_reason = detect_repetition(clean)
            if collapse_reason:
                print(f"[Abbruch] Turn {turn_index} kollabiert: {collapse_reason} "
                      f"(finish_reason={finish})", file=sys.stderr)
                # Turn trotzdem loggen für Post-mortem
                claims = extract_numeric_claims(clean)
                metrics = compute_metrics(clean, thinking, claims)
                turn = Turn(
                    index=turn_index, speaker=active_model["name"],
                    role=active_model["role"], raw_text=raw, clean_text=clean,
                    thinking_text=thinking, modified_text=clean,
                    claims=claims, metrics=metrics,
                    finish_reason=f"{finish}|collapse:{collapse_reason}",
                    timestamp=datetime.now().isoformat(),
                )
                log_file.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")
                break

            claims = extract_numeric_claims(clean)
            metrics = compute_metrics(clean, thinking, claims)
            modified = apply_escalation(clean, claims)

            turn = Turn(
                index=turn_index,
                speaker=active_model["name"],
                role=active_model["role"],
                raw_text=raw,
                clean_text=clean,
                thinking_text=thinking,
                modified_text=modified,
                claims=claims,
                metrics=metrics,
                finish_reason=finish,
                timestamp=datetime.now().isoformat(),
            )
            turns.append(turn)
            log_file.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")
            log_file.flush()

            think_info = f"think:{metrics.thinking_claim_count:>2}" if thinking else "think: –"
            finish_tag = finish if finish != "stop" else "  ok"
            print(f"[Turn {turn_index:02d}] {active_model['name']:>7} ({active_model['role']:<10}) "
                  f"| {metrics.numeric_claim_count:>2} Claims "
                  f"| {think_info} "
                  f"| {metrics.claim_density_per_100_tokens:>5.2f}/100tok "
                  f"| {finish_tag:>6} "
                  f"| {latency:>4.1f}s")
            if claims:
                preview = ", ".join(c.span for c in claims[:5])
                print(f"            → {preview}")
            if finish == "length":
                print(f"            ⚠ abgeschnitten – MAX_TOKENS erhöhen", file=sys.stderr)

    print(f"\n[Fertig] {len(turns)-1} Turns | {log_path}")
    return log_path

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Grounding Gradient MVP")
    parser.add_argument("--stufe", type=int, choices=[0, 3], default=0,
                        help="Eskalationsstufe (0 = Baseline, 3 = Inline-Markierung)")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"),
                        help="Verzeichnis für JSONL-Logs")
    args = parser.parse_args()
    run_experiment(args.stufe, args.runs_dir)
    return 0

if __name__ == "__main__":
    sys.exit(main())