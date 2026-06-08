# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

Milestones **M0–M2** are complete:
- **M1** — `src/data/` leak-free harness (`synthetic`, `windowing`, `splits`, `scaling`, `harness`, `loaders`).
- **M2** — `src/models/baselines.py` (naive + seasonal-naive), `src/models/arima.py` (AIC-grid order selection), `src/eval/metrics.py` (MASE/RMSE/MAE + `per_horizon`), `src/eval/scoreboard.py` (per-horizon comparison table).

Still to come (M3 → M8): the walk-forward backtest engine (`src/eval/walk_forward.py`), the LSTM (`src/models/lstm.py`), `src/train.py`, and `app/`. Follow the layout and build order below rather than inventing your own.

## What this project really is (read before building)

This is a portfolio project whose **point is evaluation discipline, not model accuracy**. The thesis (`SPEC.md` §0): in time series, an impressive accuracy number is almost always data leakage or a model that doesn't beat a one-line baseline. The hireable signal is leak-free splits, baselines as the bar, honest metrics, and reporting *where the LSTM loses*.

Consequences that should shape every change you make:

- **A high accuracy number is a red flag, not a win.** If results look great, suspect leakage first. Do not "improve" metrics by relaxing the evaluation.
- **The baseline is always shown.** Seasonal-naive and ARIMA are non-negotiable comparators, not optional. The LSTM only "wins" if it beats seasonal-naive on MASE per horizon — and an honest loss/tie must be reported, not hidden.
- **Three equal deliverables:** the app, `DECISIONS.md`, and a recorded whiteboard session. Code that ships without the honesty story is incomplete. Don't delete or hollow out the doc deliverables.

## Build order (non-obvious — do not reorder)

The evaluation harness and baselines are built **first, test-first**; the LSTM goes on top last. This is deliberate: the harness is the safety mechanism and the bar, and the deep model is a participant, not the hero.

1. **M1 — data + windowing + no-leakage harness (TDD).** Write `tests/test_no_leakage.py` and `tests/test_windowing.py` *before* the implementation.
2. **M2 — baselines + metrics (TDD).** naive/seasonal-naive, ARIMA, MASE/RMSE/MAE.
3. **M3 — walk-forward backtest engine (TDD).**
4. **M4 — stacked LSTM (PyTorch).** Only now. Evaluated through the *same* walk-forward harness as the baselines.
5. **M5–M8 — scoreboard/verdict → Gradio app → HF Spaces deploy → Decision Record + whiteboard.**

`test_no_leakage.py` is the crown jewel: it must fail loudly if scaling ever touches future data, if a window crosses the forecast origin, or if rows are shuffled.

## The three leakage guards (the core invariant)

Every data-handling change must preserve all three. These are what the whole project exists to demonstrate:

1. **Split by time before any scaling.** No random/shuffled splits, ever.
2. **Fit the scaler on train only**, then transform val/test forward. Never fit on the full series.
3. **Fixed forecast origin** — no window may peek past it; walk-forward uses a rolling origin and refits per fold.

If you touch windowing, scaling, or the split logic, re-run the no-leakage test and confirm it still catches violations.

## Intended architecture (per SPEC.md §6)

```
src/
  data/    loaders, seeded synthetic seasonal-demand generator, sliding-window builder (no-leakage splits)
  models/  baselines.py (naive/seasonal), arima.py, lstm.py (PyTorch stacked LSTM)
  eval/    walk_forward.py, metrics.py (MASE/RMSE/MAE per horizon), intervals.py
  train.py reproducible (seeded) training entrypoint — saves checkpoint + metrics JSON
app/app.py Gradio demo (loads a checkpoint, runs a forecast)
notebooks/ one clean, rerunnable EDA + results notebook
tests/     test_no_leakage.py, test_windowing.py, test_metrics.py
```

- **Modeling:** PyTorch (stacked LSTM) · statsmodels (ARIMA) · pandas/numpy · scikit-learn (scaling/metrics helpers).
- **Demo/host:** Gradio on Hugging Face Spaces (intentional deviation from the portfolio's Django/Render default — recorded in `DECISIONS.md`; ML demos belong on HF Spaces). Training is done offline; a small checkpoint is committed or pulled from a Release.
- **Tooling:** Python 3.12, single `pyproject.toml`, ruff + pytest, GitHub Actions CI.

## Commands

```bash
pip install -e ".[dev]"         # one-time: install runtime + dev deps (CPU torch on CI)
pytest                          # run the full test suite
pytest tests/test_no_leakage.py # run the crown-jewel leakage tests alone
pytest -k windowing             # run a single test file/function by name pattern
ruff check .                    # lint
python -m src.train             # (M4) reproducible seeded training; writes checkpoint + metrics JSON
python app/app.py               # (M6) run the Gradio demo locally
```

CI runs ruff + pytest; the no-leakage tests must pass in CI.

## Modeling decisions to make and record

These are open in the spec and must be chosen *and defended* in `DECISIONS.md` when implemented:

- **Direct vs. recursive multistep** output (recursive compounds its own error; direct trades that for more params). Pick one, record why.
- **Which real public series** is the demo primary (a public demand/retail dataset vs. a financial series via `yfinance`).
- **MASE** is the headline metric (scale-free, baseline-relative: < 1 means it beat naive); RMSE/MAE secondary; **always report per horizon**, never a single aggregate.

## Data & artifact rules

- Public + synthetic data only. **No employer data.** The synthetic generator must be seeded/reproducible.
- Do **not** commit datasets or large checkpoints — `data/`, `checkpoints/`, `*.pt`, `*.pth` are gitignored. Ship a small sample so the demo runs offline, or pull from a Release.
- Keep the **"not investment advice"** disclaimer in the README and the app footer. The financial variant is framed as *method*, never a market-beating claim.

## Source-of-truth docs

`SPEC.md` (full design + ML substance), `PLAN.md` (milestones + acceptance criteria), `DECISIONS.md` (the Situation/Decision/Risk/Change record — update it *as you build*), `WHITEBOARD-DRILL.md` (rehearsal for the recorded defense). When a design question arises, check these before guessing.
