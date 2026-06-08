# LSTM Forecaster — Implementation Plan

Companion to `SPEC.md`. The build sequence: milestones, tasks, acceptance criteria, definition of done. Self-contained.

- **Repo:** `lstm-forecaster` (public, under `cognitivefactory-hector`)
- **Approach:** build the **leak-free evaluation harness and the baselines FIRST** — they are the bar and the safety. Only then add the LSTM, and only keep it if it beats the bar honestly. The discipline is the project; the deep model is a participant, not the hero.

> **Not investment advice.** Public/synthetic data. Keep the disclaimer in the README + app footer.

---

## The spine (carry through every milestone)
Keep `DECISIONS.md` open: **Situation · Decision** (incl. rejected leaky shortcuts) **· Risk** (incl. accepted honest-but-modest accuracy) **· Change**. The hardest stance (report where the LSTM loses) is the whiteboard centerpiece — `SPEC.md` §3.

## Prerequisites
- Python 3.12, git, `gh` authenticated. A Hugging Face account for the Space (free). No API keys.

---

## Milestones

### M0 — Repo scaffold *(½ day)*
- [x] Folder + `SPEC.md` + `PLAN.md`; `README.md` (stub + disclaimer), `DECISIONS.md` (paste template from `SPEC.md` §10), `.gitignore` (Python + `data/` + `checkpoints/` + `*.pt`), `LICENSE` (MIT).
- [x] `pyproject.toml` (deps + ruff + pytest config, single source of truth); pin `torch`, `statsmodels`, `pandas`, `scikit-learn`, `gradio`, `yfinance`.
- [x] GitHub Actions: ruff + pytest.
- [x] `gh repo create … --public --push`.
- **Acceptance:** `pytest` runs (even if empty); CI green; repo on GitHub.

### M1 — Data + windowing + the no-leakage harness (TDD) *(1–2 days)* — **safety core**
- [x] `src/data/`: seeded synthetic seasonal-demand generator; a public loader (demand dataset or `yfinance`); a sliding-window builder producing multistep targets.
- [x] **Split by time before scaling; fit scaler on train only.**
- [x] **Tests first (crown jewels):** `test_no_leakage.py` — scaler params depend only on train; no window crosses the forecast origin; no row shuffling. `test_windowing.py` — shapes/targets correct.
- **Acceptance:** `pytest` green; the leakage test fails loudly if you "accidentally" scale on the full series.

### M2 — Baselines + metrics (TDD) *(1 day)* — **the bar**
- [x] `src/models/baselines.py`: naive + seasonal-naive. `src/models/arima.py`: ARIMA (order via AIC / small grid).
- [x] `src/eval/metrics.py`: MASE (vs. naive), RMSE, MAE; per-horizon aggregation.
- [x] Tests: MASE of the naive model ≈ 1.0; metrics match hand-computed values on a fixture.
- **Acceptance:** `pytest` green; baseline scoreboard prints per horizon. (`src/eval/scoreboard.py`)

### M3 — Walk-forward backtest engine (TDD) *(1 day)*
- [x] `src/eval/walk_forward.py`: rolling origin; refit per fold; aggregate metrics across folds; prediction intervals (`src/eval/intervals.py`).
- [x] Tests: folds never use future data; horizon-error increases monotonically on a constructed ramp.
- **Acceptance:** `pytest` green; baselines evaluated via walk-forward.

### M4 — Stacked LSTM (PyTorch) *(2 days)*
- [ ] `src/models/lstm.py`: stacked LSTM, hidden-state sizing, inter-layer dropout, **direct multi-horizon** output (or recursive — record the choice).
- [ ] `src/train.py`: reproducible (seeded) training; early stopping on a time-ordered val fold; save checkpoint + metrics JSON.
- [ ] Tests: deterministic forward pass under a seed; output shape = horizon; overfits a tiny synthetic set (sanity).
- **Acceptance:** LSTM trains, evaluated through the *same* walk-forward harness as the baselines.

### M5 — Honest scoreboard + verdict *(½ day)*
- [ ] Compare LSTM vs. baselines per series, per horizon; write the plain verdict (incl. any losses).
- [ ] Plots: forecast vs. actual w/ baseline overlay; error-by-horizon.
- **Acceptance:** a results table + verdict exist; at least one honest "baseline wins / ties" is reported if true.

### M6 — Gradio app *(1 day)*
- [ ] `app/app.py`: dataset selector → run → forecast plot + scoreboard + verdict; footer disclaimer.
- **Acceptance:** app runs locally; demo viewer can drive it.

### M7 — Deploy to Hugging Face Spaces + README *(½–1 day)*
- [ ] Push to a Gradio **HF Space** (commit a small checkpoint or pull from a Release).
- [ ] `README.md`: what/why, one-command run, data provenance + limits, links to Space + `DECISIONS.md` + whiteboard.
- **Acceptance:** public Space URL works; the GIF in the README matches reality.

### M8 — Decision Record + Whiteboard session *(½ day)* — **the differentiator, don't skip**
- [ ] Complete `DECISIONS.md`; record the 5–8 min session using `SPEC.md` §3.1 — center #1 (prove no leakage) and #6 (where it loses).
- [ ] Embed/link on hector-garza.com.
- **Acceptance:** a stranger can read `DECISIONS.md` + watch the video and explain why your accuracy is trustworthy.

---

## Testing strategy
- **The no-leakage harness, windowing, and metrics are the crown jewels — test them hardest and first.**
- LSTM gets sanity tests (deterministic forward pass, overfit-tiny-set), not accuracy assertions.
- The app/notebook are demonstrated by the recording.

## Risk register (project execution)
| Risk | Mitigation |
|---|---|
| Data leakage inflates results | No-leakage test is mandatory and runs in CI; train-only scaling; no shuffling. |
| LSTM doesn't beat the baseline | That's a *finding*, not a failure — report it; it's the honesty signal. |
| Over-claiming on financial series | Frame as method, not market-beating; forecast demand/vol, disclaimer. |
| Huge checkpoints/datasets in git | Gitignored; small sample in-repo or a Release. |
| Skipping M8 | M8 *is* the portfolio — the "trust me with time data" story. |

## Definition of Done
See `SPEC.md` §8 — app + decision record + whiteboard, all linked; the no-leakage test passes in CI; the baseline is always shown and honest losses are reported.
