# LSTM Forecaster

Multistep demand & financial forecasting with a stacked PyTorch LSTM — measured honestly against classic baselines (naive, ARIMA) under strict, leak-free, time-ordered validation, so the deep model only gets credit when it actually earns it.

> **Status:** scaffolded (spec + plan in place). Build follows `PLAN.md` (M0 → M8).
> **Not investment advice.** Public + synthetic data only; not connected to any employer system.

Part of [hector-garza.com](https://hector-garza.com)'s portfolio. One of three equal deliverables: the app, a **Decision Record** ([`DECISIONS.md`](./DECISIONS.md)), and a recorded whiteboard session. A working model no longer proves competence — the judgment behind it does. See [`SPEC.md`](./SPEC.md) §0.

## What it does
- Forecasts a series **multiple steps ahead** with a stacked LSTM.
- **Always** compares against seasonal-naive and ARIMA baselines.
- **Leak-free evaluation:** time-ordered walk-forward, train-only scaling, fixed forecast origin.
- Honest scoreboard: **MASE** (vs. naive), RMSE, MAE — per model, per horizon — plus prediction intervals and a plain verdict (including where the LSTM *loses*).

## Tech stack
- **Modeling:** PyTorch (stacked LSTM) · statsmodels (ARIMA) · pandas / numpy / scikit-learn
- **Demo:** Gradio · **Host:** Hugging Face Spaces
- **Tooling:** Python 3.12 · `pyproject.toml` · ruff + pytest · GitHub Actions CI

> Stack note: this is a PyTorch/ML project, so it deviates from the portfolio's Django/Render default on purpose — **Gradio + Hugging Face Spaces** is where ML demos live and signals ecosystem fluency. The [hector-garza.com](https://hector-garza.com) hub links out to the Space.

## Links (filled in as the build progresses)
- 🔗 Live demo (HF Space): _TBD_
- 🧠 Decision record: [`DECISIONS.md`](./DECISIONS.md)
- 🎥 Whiteboard walkthrough: _TBD_

## Build
See [`PLAN.md`](./PLAN.md) — M0 (scaffold) → M8. The leak-free evaluation harness and the baselines are built **first** (test-first); the LSTM goes on top and only keeps its place if it beats the bar honestly.
