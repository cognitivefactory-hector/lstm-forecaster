"""Gradio demo: pick a dataset, run the leak-free backtest, read the verdict.

Thin UI over ``app.pipeline.run_evaluation``. The viewer sees the forecast vs.
actual (baseline overlaid), MASE by horizon, the plain-language verdict — losses
included — and the per-horizon scoreboard. The whole point: a forecast you can
trust because the evaluation can't see the future, stated honestly.
"""

from __future__ import annotations

import gradio as gr

from app.datasets import dataset_names, get_dataset
from app.pipeline import run_evaluation

# Demo training config — enough epochs (with early stopping) that the LSTM gets
# a competent, fair fight against the baselines before the verdict is read.
_DEMO = dict(
    horizon=7,
    input_len=28,
    epochs=400,
    lr=5e-3,
    patience=30,
    hidden_size=48,
    num_layers=2,
)

_INTRO = """# LSTM Forecaster — honest time-series evaluation

A stacked PyTorch LSTM measured against naive & seasonal-naive baselines under
**leak-free, time-ordered walk-forward** validation. The deep model only gets
credit when it beats the baseline on **MASE** (scale-free: < 1 beats naive) —
**per horizon**, and the verdict says plainly where it *loses*.
"""

_FOOTER = (
    "Illustrative tool on public + synthetic data. The financial series forecasts "
    "**volatility, not price** — *not investment advice*, not connected to any "
    "employer system."
)


def run_demo(dataset_name: str):
    """Handler: evaluate the chosen dataset and return the demo artifacts."""
    spec = get_dataset(dataset_name)
    result = run_evaluation(spec.series, season=spec.season, **_DEMO)
    verdict_md = f"### Verdict\n```\n{result.verdict.text}\n```\n_{spec.note}_"
    return result.forecast_fig, result.error_fig, verdict_md, result.board


def build_demo() -> gr.Blocks:
    """Assemble the Gradio interface."""
    with gr.Blocks(title="LSTM Forecaster") as demo:
        gr.Markdown(_INTRO)
        with gr.Row():
            dataset = gr.Dropdown(
                choices=dataset_names(),
                value="synthetic_demand",
                label="Dataset",
            )
            run = gr.Button("Run backtest", variant="primary")
        with gr.Row():
            forecast_plot = gr.Plot(label="Forecast vs. actual (baseline overlaid)")
            error_plot = gr.Plot(label="MASE by horizon")
        verdict_md = gr.Markdown()
        scoreboard = gr.Dataframe(label="Per-horizon scoreboard")
        gr.Markdown(_FOOTER)

        run.click(
            run_demo,
            inputs=[dataset],
            outputs=[forecast_plot, error_plot, verdict_md, scoreboard],
        )
    return demo


def main() -> None:
    build_demo().launch()


if __name__ == "__main__":
    main()
