# Decision Record — LSTM Forecaster

The four questions that make judgment portable. These are **first-draft answers** (from `SPEC.md` §1.1) — pressure-test and revise them in the recorded whiteboard session, then keep what survives.

## Situation
Demand and financial series are noisy and non-stationary. The temptation is to throw a deep LSTM at one and report a great-looking fit — but most such results are **data leakage** (scaler fit on the full series, shuffled splits, peeking past the forecast origin) or a model that **doesn't beat a naive baseline.** Facts I have: historical sequential data. Facts I'm missing: any guarantee the evaluation is honest, and any *a priori* reason the series is predictable at all.

## Decision
A **stacked LSTM for multistep forecasting**, **always alongside classic baselines (seasonal-naive, ARIMA)**, evaluated with **strict time-ordered walk-forward splits, train-only scaling, and a fixed forecast origin** — no leakage — reporting **whether the LSTM actually beats the baseline, per horizon.**
**Rejected:** a single impressive in-sample fit; fitting the scaler on the whole series; shuffling time-series rows.

## Risk
The killers: **silent data leakage** producing a fantasy accuracy that collapses live, and **overfitting** a complex model that's worse than one line of code. Mitigations: walk-forward validation, train-only scaling, baselines as the bar, scale-free metrics (MASE), prediction intervals, error-by-horizon.
**Consciously accepted:** a modest, honest accuracy over an impressive fake one — and that on some series the LSTM ties or loses to ARIMA, which I will *say plainly.*

## Change
Forecasts I can trust because the evaluation can't see the future; a clear statement of *when* the deep model is worth its complexity and when a baseline wins. The prevented loss: deploying a leaky model that dazzles offline and fails in production.

## Whiteboard session
- Recording: _TBD_
- The leakage trap I avoided: _…_
- Where the LSTM loses to the baseline: _…_

---

## Engineering decisions (recorded as built)
- **Modeling:** PyTorch stacked LSTM; statsmodels ARIMA; seasonal-naive baseline as the bar. Direct multi-horizon output (or recursive — record the choice).
- **Evaluation = the safety core, built first:** time-ordered walk-forward, **train-only scaling**, fixed origin; an explicit no-leakage test runs in CI.
- **Stack deviation (on purpose):** PyTorch + **Gradio + Hugging Face Spaces** instead of the portfolio's Django/Render — ML demos belong on HF Spaces and it signals ecosystem fluency. _Why recorded:_ consistency matters for web apps; the right *tool* matters for ML.
- **Artifacts:** don't commit datasets/large checkpoints — small sample in-repo or a Release.
