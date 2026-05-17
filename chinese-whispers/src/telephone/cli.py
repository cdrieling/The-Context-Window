import logging
import os
import sys

import click
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@click.group()
def cli():
    """Telephone Game — LLM paraphrase drift runner."""


@cli.command()
def probe():
    """Connect to the oMLX server and list available models."""
    from telephone.inference import make_client, probe_models

    client = make_client()
    models = probe_models(client)
    click.echo("Available models on server:")
    for m in models:
        click.echo(f"  {m}")
    if not models:
        click.echo("  (none — is the server running?)")


@cli.command()
@click.option("--experiment", "-e", required=True, help="Experiment name from experiments.yaml")
@click.option("--seed", "-s", type=int, default=None, help="Override seed (single run)")
@click.option("--rounds", "-r", type=int, default=10, help="Number of rounds (default: 10)")
def run(experiment: str, seed: int | None, rounds: int):
    """Run a telephone game experiment."""
    from telephone.runner import load_config, run_single

    _, experiments_cfg = load_config()
    if experiment not in experiments_cfg:
        click.echo(f"Unknown experiment '{experiment}'.", err=True)
        sys.exit(1)

    seeds = [seed] if seed is not None else experiments_cfg[experiment]["seeds"]
    click.echo(f"Experiment: {experiment} | seeds: {seeds} | rounds: {rounds}")

    for s in seeds:
        click.echo(f"\n=== Seed {s} ===")
        summary = run_single(experiment, seed=s, rounds=rounds)
        click.echo(f"Summary: {summary}")
