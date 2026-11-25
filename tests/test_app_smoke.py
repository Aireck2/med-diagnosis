"""Integration smoke tests for the Streamlit app using streamlit.testing.v1.AppTest."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as _go  # Pre-warm Plotly before LocalScriptRunner timer
import streamlit as _st  # Pre-warm Streamlit runtime modules
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
_ = (_go, _st)


def test_app_boots_and_renders_default_guided_sequence_without_errors() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    assert not at.exception
    assert len(at.metric) == 4
    assert "1.50 %" in at.metric[0].value


def test_app_switches_presets_and_commutativity_overlay_cleanly() -> None:
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception

    options = list(at.sidebar.selectbox[0].options)
    for option in options[1:]:
        at.sidebar.selectbox[0].select(option).run()
        assert not at.exception

    # Toggle commutativity overlay checkbox
    assert len(at.checkbox) == 1
    at.checkbox[0].check().run()
    assert not at.exception
    assert len(at.success) == 1
