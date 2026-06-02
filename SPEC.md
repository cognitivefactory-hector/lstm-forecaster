# LSTM Forecaster — Design Spec

**Project 7 of the Hector Garza portfolio.** Self-contained: everything needed to start this as its own repository is in this file and its companion `PLAN.md`. You do not need any other file from the `career/` folder to build this.

- **Owner:** Hector Garza · hectorg@smartxchain.com · hector-garza.com
- **Status:** Spec — ready to build
- **Suggested repo name:** `lstm-forecaster`
- **One-liner:** Multistep demand & financial forecasting with a stacked PyTorch LSTM — measured honestly against classic baselines (naive, ARIMA) under strict, leak-free, time-ordered validation, so the deep model only gets credit when it actually earns it.

> **Illustrative tool on public + synthetic data.** Not investment advice; not connected to any employer system.

---

## 0. Read this first — what this project is *really* for

This is a job-search portfolio project to **demonstrate PyTorch and sequence-modeling competence** — but it is **not** a "my LSTM got 95% accuracy" demo. In time series, an impressive accuracy number is almost always **data leakage** or a model that doesn't actually beat a one-line baseline. The hireable signal is the **discipline**: leak-free splits, baselines as the bar, honest metrics, and the intellectual honesty to report when the deep model *loses*.

Three deliverables of equal weight:
1. **The working app** (hosted, clickable) — forecasts vs. actuals with the baseline overlaid.
2. **A Decision Record** (`DECISIONS.md`) structured around the four questions below.
3. **A recorded whiteboard session** (5–8 min) defending your evaluation against "prove it's not leakage."

A hiring manager who opens this repo should learn you can be *trusted with time-dependent data* — the rarest thing in applied ML.

---

## 1. The spine — four questions that make judgment portable

> **1 · Situation** — What's happening, who's involved, the constraints, the facts you have and the facts that are *missing*. Context is where judgment begins.
>
> **2 · Decision** — The plausible paths, the one you took, and the credible options you *rejected*. Rejection shows what you refused to hand-wave.
>
> **3 · Risk** — What could go wrong, what you removed, and what you *consciously accepted*. Prevented losses count.
>
> **4 · Change** — What's different now. Connect the judgment to a real change in the work.

### 1.1 First-draft answers for LSTM Forecaster (defend/revise on camera)

- **Situation.** Demand and financial series are noisy and non-stationary. The temptation is to throw a deep LSTM at one and report a great-looking fit — but most such results are **data leakage** (scaler fit on the full series, shuffled splits, peeking past the forecast origin) or a model that **doesn't beat a naive baseline.** Facts I have: historical sequential data. Facts I'm missing: any guarantee the evaluation is honest, and any *a priori* reason the series is predictable at all.
- **Decision.** Build a **stacked LSTM for multistep forecasting**, but **always alongside classic baselines (seasonal-naive, ARIMA)**, and evaluate with **strict time-ordered walk-forward splits, train-only scaling, and a fixed forecast origin** — no leakage. Report **whether the LSTM actually beats the baseline**, per horizon. **Rejected:** a single impressive in-sample fit; fitting the scaler on the whole series (a classic leak); shuffling time-series rows.
- **Risk.** The killers: **silent data leakage** that yields a fantasy accuracy which collapses live, and **overfitting** a complex model that's worse than one line of code. Mitigations: walk-forward validation, train-only scaling, baselines as the bar, scale-free honest metrics (MASE alongside RMSE/MAE), prediction intervals, and error-by-horizon. **Consciously accepted:** a modest, honest accuracy over an impressive fake one — and that on some series the LSTM ties or loses to ARIMA, which I will *say plainly.*
- **Change.** Forecasts you can trust because the evaluation can't see the future; a clear statement of *when* the deep model is worth its complexity and when a baseline wins. Prevented loss: deploying a leaky model that dazzles offline and fails in production.

---

## 2. Why this project (market fit)

- Time-series forecasting is everywhere in your target space: **demand planning, supply/consumable forecasting, predictive maintenance**, and your own **quant** work. It threads into PlantGPT (demand drives scheduling) and Options Lab (financial series).
- **PyTorch + sequence modeling** is a directly-screened skill; the no-leakage discipline is what separates someone who's shipped time-series from someone who's only done tutorials.
- The honesty angle (deep model vs. baseline) is the rarest and most senior signal — and it's the judgment thesis applied to ML.

---

## 3. The staged whiteboard session (recorded deliverable)

**Format.** 5–8 min, screen + voice, defending the design against push-back. Use a strong ML-literate friend or answer the script below on camera. Preserve what survives in `DECISIONS.md`.

### 3.1 Adversarial challenge script
1. **"Your LSTM looks great — prove it isn't data leakage."** *(Walk-forward, train-only scaling, no shuffling, fixed forecast origin.)*
2. **"A naive 'tomorrow = today' (or seasonal-naive) baseline is probably as good. Why the LSTM?"** *(Show MASE vs. baseline; admit where it doesn't win.)*
3. **"Stock prices are ~a random walk. Forecasting them is a fool's errand — defend the financial variant."** *(Honesty: forecast demand/volatility, not price level; what's actually predictable.)*
4. **"Your metric flatters you. Why MASE and these splits?"** *(Scale-free, baseline-relative; limits.)*
5. **"Multistep error compounds. Direct or recursive? Show error by horizon."**
6. **"Show me where your model LOSES to the baseline."** *(The honest negative result — the thesis analog of "the trade you didn't take.")*

### 3.2 What the recording must show
- The **Situation → Decision → Risk → Change** arc, in your words.
- A concrete **leakage trap you avoided** (and how you'd detect it).
- At least one **honest loss** to the baseline, or a crisp reason the LSTM wins.
- A pointer to `DECISIONS.md`.

---

## 4. Product specification

### 4.1 Users
- **Primary:** you / an analyst comparing forecasting approaches on a series.
- **Demo viewer:** a hiring manager who must see, in 60 seconds, forecast-vs-actual with the baseline overlaid and the honest metrics.

### 4.2 Core features (MVP)
1. **Pick a dataset:** a synthetic seasonal-demand series (seeded) and at least one real public series (e.g., retail demand, or a financial series via `yfinance`).
2. **Models:** seasonal-naive baseline, ARIMA (statsmodels), and a **stacked LSTM** (PyTorch) for **multistep** forecasting.
3. **Leak-free evaluation:** time-ordered **walk-forward** backtest; train-only scaling; configurable horizon.
4. **Honest scoreboard:** MASE (vs. naive), RMSE, MAE — **per model, per horizon**; prediction intervals.
5. **Plots:** forecast vs. actual with the baseline overlaid; error-by-horizon; the held-out windows.
6. **A plain verdict:** "LSTM beats baseline by X% MASE on series A; ties on B" — stated, not hidden.

### 4.3 Screens / UI
- A **Gradio** app: dataset selector → run → forecast plot + scoreboard + the verdict. (Demo viewer drives it.)

### 4.4 Explicit non-goals (YAGNI)
- No live/real-time data feeds or auto-trading.
- No transformer/foundation-model forecasting (note as an extension) — the point is mastering LSTMs *honestly*.
- No hyperparameter mega-search; a small, documented search is enough.
- No claim of beating the market — the financial variant is about *method*, framed honestly.

---

## 5. Data (public + synthetic — be honest about limits)

- **Synthetic seasonal demand:** a seeded generator (trend + seasonality + noise + occasional shocks) — fully reproducible, obviously synthetic.
- **Real public series:** a public demand/retail dataset and/or a financial series via `yfinance` (keyless). Cache locally; ship a small sample so the demo runs offline.
- **No employer data.** State data provenance and limitations in the README.

---

## 6. Architecture & stack

A deliberate deviation from the portfolio's Django/Render default — ML modeling + an ML-idiomatic demo/host. **Recorded as a decision** in `DECISIONS.md`.

```
repo
├── src/
│   ├── data/        loaders, synthetic generator, windowing (no-leakage splits)
│   ├── models/      baselines.py (naive/seasonal), arima.py, lstm.py (PyTorch)
│   ├── eval/        walk_forward.py, metrics.py (MASE/RMSE/MAE), intervals.py
│   └── train.py     reproducible training entrypoint (saves checkpoints + metrics)
├── app/  app.py     Gradio demo (loads a checkpoint, runs a forecast)
├── notebooks/       one clean, rerunnable EDA + results notebook
└── tests/           test_no_leakage.py, test_windowing.py, test_metrics.py
```

- **Modeling:** **PyTorch** (stacked LSTM) · **statsmodels** (ARIMA) · **pandas/numpy** · **scikit-learn** (scaling/metrics helpers).
- **Demo:** **Gradio**. **Host:** **Hugging Face Spaces** (free, ML-idiomatic; CPU is fine for inference). Linked from hector-garza.com.
- **Tooling:** Python 3.12, single `pyproject.toml`, **ruff + pytest**, GitHub Actions CI (matches the portfolio's conventions).

---

## 7. ML substance (get it right — you'll be asked)

- **Windowing:** sliding input windows → multistep targets; **split by time** before any scaling; **fit the scaler on train only**, transform val/test.
- **LSTM:** stacked layers, hidden state sizing, dropout between layers; **direct** multi-horizon output (or recursive — pick and defend); teacher-forcing only if justified.
- **Baselines:** seasonal-naive (the bar) and ARIMA (order chosen by AIC or a small grid) — non-negotiable comparators.
- **Walk-forward:** rolling origin; refit/refresh per fold; aggregate metrics across folds.
- **Metrics:** **MASE** (scale-free, baseline-relative) headline; RMSE/MAE secondary; report **per horizon**; prediction intervals (quantile or residual-based).
- **Leakage audit:** an explicit test that scaling/feature construction never uses future data; no row shuffling.

---

## 8. Definition of Done

- [ ] **App** on a public URL (HF Space): pick a dataset → see forecast vs. actual with baseline overlaid + the MASE/RMSE scoreboard per horizon + the plain verdict.
- [ ] **Honesty visible:** the baseline is always shown; at least one series where the LSTM does **not** clearly win is reported, not hidden.
- [ ] **`README.md`** — what/why, one-command local run, data provenance + limits, links to live demo + `DECISIONS.md` + whiteboard video, and the not-advice disclaimer.
- [ ] **`DECISIONS.md`** — the §1 template completed (rejected the leaky single-fit; accepted modest-but-honest accuracy).
- [ ] **Whiteboard recording** linked from README and on hector-garza.com.
- [ ] **Tests pass:** the no-leakage test, windowing, and metrics (see `PLAN.md`).

---

## 9. Hosting / deployment
- **Hugging Face Spaces** (Gradio SDK) — push the repo, the Space builds it. Free CPU tier is fine for inference; training is done offline and a checkpoint is committed (or pulled from a release).
- Link from hector-garza.com; optionally embed the Space.
- (Render is an alternative, but HF Spaces is the ML-idiomatic venue and signals ecosystem fluency.)

---

## 10. Repo bootstrap (how to start this as its own repo)

```bash
mkdir lstm-forecaster && cd lstm-forecaster
cp /path/to/07-lstm-forecaster/SPEC.md .
cp /path/to/07-lstm-forecaster/PLAN.md .
# seed: README.md, DECISIONS.md (template below), .gitignore (python + data/ + checkpoints), LICENSE (MIT)

git init && git add -A && git commit -m "chore: scaffold lstm-forecaster (spec + plan)"
git branch -M main
gh repo create cognitivefactory-hector/lstm-forecaster --public --source=. --remote=origin --push
```

> PUBLIC repo. **Don't commit datasets or large checkpoints** (gitignored; use a small sample or a GitHub Release). No API keys needed.

### `DECISIONS.md` starter
```markdown
# Decision Record — LSTM Forecaster

## Situation
<noisy non-stationary series; leakage/overfitting temptation; missing: honest eval + a-priori predictability>

## Decision
<stacked LSTM + naive/ARIMA baselines; walk-forward, train-only scaling, fixed origin; report per-horizon win/lose; REJECTED single in-sample fit, full-series scaling, shuffling>

## Risk
<silent leakage; overfit worse-than-baseline; walk-forward + MASE-vs-baseline + intervals + per-horizon error; ACCEPTED honest-but-modest accuracy, and reporting losses>

## Change
<trustworthy forecasts; clear when deep is worth it; prevented a leaky model that fails live>

## Whiteboard session
- Recording: <link>
- The leakage trap I avoided: <…>
- Where the LSTM loses to the baseline: <…>
```

---

## 11. Open questions to resolve in the plan
- Direct vs. recursive multistep (pick + defend).
- Which real series for the demo (demand dataset vs. financial via yfinance) — pick one primary.
- Ship the trained checkpoint in-repo (small) vs. via a GitHub Release.
