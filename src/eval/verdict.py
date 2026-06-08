"""The plain-language verdict — where the deep model wins, ties, and LOSES.

This is the honesty centerpiece (SPEC §0, whiteboard Q6). It reads the
per-horizon scoreboard and states, in words, whether the model beats the
baseline overall and at which horizons it does not. Losses are named explicitly,
never hidden — knowing *when not to reach for the deep model* is the judgment the
project exists to demonstrate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Verdict:
    model: str
    baseline: str
    model_mean_mase: float
    baseline_mean_mase: float
    overall: str  # "wins" | "ties" | "loses"
    wins: list[int] = field(default_factory=list)  # horizons model beats baseline
    losses: list[int] = field(default_factory=list)  # horizons baseline wins
    ties: list[int] = field(default_factory=list)
    text: str = ""


def _outcome(model_value: float, baseline_value: float, tie_rel: float) -> str:
    """Classify model vs baseline as win/tie/loss within a relative tolerance."""
    ratio = model_value / baseline_value
    if ratio < 1.0 - tie_rel:
        return "win"
    if ratio > 1.0 + tie_rel:
        return "loss"
    return "tie"


def build_verdict(
    board: pd.DataFrame,
    *,
    model: str,
    baseline: str,
    metric: str = "mase",
    tie_rel: float = 0.05,
) -> Verdict:
    """Compare ``model`` against ``baseline`` per horizon and write the verdict.

    Args:
        board: Per-(model, horizon) scoreboard from ``build_scoreboard``.
        model: The model under test (e.g. ``"lstm"``).
        baseline: The reference baseline (e.g. ``"seasonal_naive"``).
        metric: Scoreboard column to judge on (MASE by default).
        tie_rel: Relative band within which a difference counts as a tie.

    Raises:
        ValueError: if ``model`` or ``baseline`` is absent from the board.
    """
    present = set(board["model"].unique())
    for name in (model, baseline):
        if name not in present:
            raise ValueError(f"{name!r} not in scoreboard (have {sorted(present)})")

    m = board[board["model"] == model].set_index("horizon")[metric]
    b = board[board["model"] == baseline].set_index("horizon")[metric]

    wins, losses, ties = [], [], []
    for h in sorted(m.index):
        outcome = _outcome(m[h], b[h], tie_rel)
        {"win": wins, "loss": losses, "tie": ties}[outcome].append(int(h))

    model_mean = float(m.mean())
    baseline_mean = float(b.mean())
    overall = {
        "win": "wins",
        "loss": "loses",
        "tie": "ties",
    }[_outcome(model_mean, baseline_mean, tie_rel)]

    verdict = Verdict(
        model=model,
        baseline=baseline,
        model_mean_mase=model_mean,
        baseline_mean_mase=baseline_mean,
        overall=overall,
        wins=wins,
        losses=losses,
        ties=ties,
    )
    verdict.text = _render(verdict, metric)
    return verdict


def _render(v: Verdict, metric: str) -> str:
    pct = (v.model_mean_mase / v.baseline_mean_mase - 1.0) * 100.0
    headline = {
        "wins": f"{v.model} BEATS {v.baseline}",
        "ties": f"{v.model} TIES {v.baseline}",
        "loses": f"{v.model} LOSES to {v.baseline}",
    }[v.overall]
    lines = [
        f"VERDICT: {headline} overall "
        f"(mean {metric.upper()} {v.model_mean_mase:.3f} vs {v.baseline_mean_mase:.3f}, "
        f"{pct:+.1f}%).",
    ]
    if v.wins:
        lines.append(f"  Wins at horizon(s): {v.wins}.")
    if v.losses:
        lines.append(f"  Loses at horizon(s): {v.losses}.")
    if v.ties:
        lines.append(f"  Ties at horizon(s): {v.ties}.")
    return "\n".join(lines)
