"""Hugging Face Spaces entrypoint.

Named distinctly from the ``app/`` package to avoid import ambiguity. HF Spaces
(configured via the README front-matter ``app_file: space_app.py``) imports this
module and launches ``demo``. The Blocks is built at import time; no training
runs until the viewer clicks "Run backtest".
"""

from app.app import build_demo

demo = build_demo()

if __name__ == "__main__":
    demo.launch()
