"""Test the Hugging Face Spaces entrypoint exposes a launchable demo — M7."""

import gradio as gr

import space_app


def test_exposes_a_launchable_blocks_demo():
    # HF Spaces imports this module and launches `demo`; it must be a Blocks
    # built at import time (no network, no training until the user clicks Run).
    assert isinstance(space_app.demo, gr.Blocks)
