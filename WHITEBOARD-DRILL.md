# Whiteboard Drill — LSTM Forecaster (design-stage)

> Rehearsal for the recorded whiteboard session. **The push** is me playing tough reviewer; **Defense** is the position that survives; **⚠ Your move** is what only you can answer once you've built/measured it. Fold the survivors into `DECISIONS.md`, then record.
> Scope: design-stage. Re-run after **M5** with the real per-horizon scoreboard (incl. any losses) in hand.

## Q1 (the killer) — "Your LSTM looks great — prove it isn't data leakage."
**The push:** Impressive time-series accuracy is almost always a leak.
**Defense (survives):** Three guards, all testable: **walk-forward** (rolling origin, never a single in-sample fit), **scaler fit on train only** then applied forward, and **no row shuffling / fixed forecast origin**. There's an explicit `test_no_leakage` in CI that fails if scaling ever touches future data. If accuracy depends on any of those being violated, it's not accuracy.
**⚠ Your move:** Show the leakage test in CI and the out-of-sample vs. in-sample gap.

## Q2 — "A naive 'tomorrow = today' (or seasonal-naive) baseline is probably as good. Why the LSTM?"
**The push:** You added deep learning for nothing.
**Defense (survives):** That's exactly why the baseline is **built first and always shown** — the LSTM only earns its place if it beats seasonal-naive on **MASE**, per horizon. If it doesn't, I report that. The headline isn't "LSTM wins," it's "here's whether the complexity is justified."
**⚠ Your move:** Have the MASE-vs-baseline table ready; if the LSTM ties/loses on a series, say so plainly (that honesty is the signal).

## Q3 — "Stock prices are ~a random walk. The financial variant is a fool's errand."
**The push:** You can't forecast price.
**Defense (survives):** Agreed on price level — so the financial variant targets what's more forecastable (volatility / demand-like series), and I frame it as *method*, not a market-beating claim. For the random-walk series, the naive baseline is genuinely hard to beat — and reporting that the LSTM *can't* beat it is itself the correct, honest result.

## Q4 — "Your metric flatters you. Why MASE and these splits?"
**The push:** You chose a metric that looks good.
**Defense (survives):** **MASE** is scale-free and explicitly **baseline-relative** (MASE < 1 means you beat naive) — it's hard to flatter yourself with. RMSE/MAE are secondary and reported per horizon. The splits are time-ordered walk-forward, the opposite of a flattering random split.
**⚠ Your move:** Be ready to state MASE's limits (it assumes the naive baseline is a fair reference).

## Q5 — "Multistep error compounds. Direct or recursive? Show error by horizon."
**The push:** Your h=1 number hides h=12 garbage.
**Defense (survives):** I report **error by horizon**, not a single aggregate, so the degradation is visible. I chose [direct multi-horizon / recursive — state which] because [reason]; recursive compounds its own errors, direct trades that for more parameters.
**⚠ Your move:** Lock direct vs. recursive and defend it; show the error-by-horizon curve.

## Q6 — "Show me where your model LOSES to the baseline."
**The push:** Anyone can cherry-pick a series where it wins.
**Defense (survives):** Here's a series where seasonal-naive ties or beats the LSTM — and I keep it in the report rather than hiding it. Knowing *when not to reach for the deep model* is the judgment; a portfolio that only shows wins is the thing that doesn't carry signal anymore.
**⚠ Your move:** Make sure at least one honest loss/tie is in the final scoreboard.

## Verdict — SDRC after the drill
- **Holds:** leak-free discipline; baselines-as-the-bar; report-the-losses honesty.
- **Sharpen:** lead with **Q1** (leakage proof) and **Q6** (the honest loss); lock direct-vs-recursive (Q5); state MASE's limits (Q4).
- **Land this line in the room:** *"The hard part of time-series ML isn't the model — it's an evaluation that can't see the future, and the honesty to say when a one-line baseline wins."*
