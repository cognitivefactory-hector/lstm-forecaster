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
- **Modeling:** PyTorch stacked LSTM; statsmodels ARIMA; seasonal-naive baseline as the bar.
- **Multistep output: DIRECT, not recursive (locked in M4).** The LSTM emits all `horizon` steps at once from the input window. _Why:_ recursive forecasting feeds its own predictions back and compounds error across the horizon; direct trades that for more output parameters and matches the direct-target windowing already in `src/data/windowing.py`. This is the answer to whiteboard Q5.
- **Honest finding (M4): on the seeded synthetic period-7 series, the LSTM does *not* beat seasonal-naive** (mean MASE ≈ 1.25 vs ≈ 1.06 through the walk-forward harness), though it edges ahead at one horizon. A pure-seasonal series is exactly where a one-line seasonal baseline is hard to beat; reporting this rather than tuning until the deep model "wins" is the point (SPEC §0, whiteboard Q6).
- **Demo dataset (M6): real public series = AAPL realized volatility (yfinance).** Forecast *volatility* (rolling std of daily log returns), not price level — price is ~a random walk where naive is unbeatable for the right reasons (whiteboard Q3). A small real sample is committed (`app/samples/aapl_close.csv`) so the Gradio demo runs offline. _Rejected:_ a demand/retail CSV (less tied to the quant narrative) and synthetic-only (no real-data test).
- **Honest finding (M6): the LSTM loses to the persistence baseline on volatility too** (e.g. MASE ≈ 1.5 at h=1 vs naive ≈ 0.58), because realized volatility is smooth and highly persistent — a one-line "tomorrow ≈ today" is genuinely hard to beat, and MASE amplifies small absolute errors against a tiny day-to-day-change denominator. This is the whiteboard Q3/Q4 story, told straight. **Open portfolio decision (see handoff): both demo series currently favor simple baselines; consider enriching the synthetic generator (multi-seasonality / shocks per SPEC §5) so the deep model gets a series where it can legitimately add value — without rigging the result.**
- **Evaluation = the safety core, built first:** time-ordered walk-forward, **train-only scaling**, fixed origin; an explicit no-leakage test runs in CI.
- **Stack deviation (on purpose):** PyTorch + **Gradio + Hugging Face Spaces** instead of the portfolio's Django/Render — ML demos belong on HF Spaces and it signals ecosystem fluency. _Why recorded:_ consistency matters for web apps; the right *tool* matters for ML.
- **Artifacts:** don't commit datasets/large checkpoints — small sample in-repo or a Release.
