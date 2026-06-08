"""Tests for the Gradio app assembly — M6.

The heavy logic lives in app.pipeline (tested there); here we just confirm the
UI constructs and exposes both datasets, so the demo viewer has something to
drive. The full click-through is verified by launching it locally.
"""

import gradio as gr

from app.app import build_demo


def test_build_demo_returns_a_gradio_blocks():
    demo = build_demo()
    assert isinstance(demo, gr.Blocks)


def test_demo_offers_both_datasets_in_a_dropdown():
    demo = build_demo()
    dropdowns = [b for b in demo.blocks.values() if isinstance(b, gr.Dropdown)]
    assert dropdowns, "expected a dataset dropdown"
    choices = {value for d in dropdowns for (_, value) in d.choices}
    assert {"synthetic_demand", "aapl_volatility"} <= choices
