"""Pruebas de integración de la interfaz Streamlit usando streamlit.testing.v1.AppTest."""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as _go  # Precargar Plotly antes del temporizador de LocalScriptRunner
import streamlit as _st  # Precargar módulos del entorno de ejecución de Streamlit
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")
_ = (_go, _st)


def test_app_boots_and_renders_default_guided_sequence_without_errors() -> None:
    """Verifica que la aplicación arranque y renderice el protocolo guiado sin excepciones."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()

    assert not at.exception
    assert len(at.metric) == 4
    assert "1.50 %" in at.metric[0].value


def test_app_switches_presets_and_commutativity_overlay_cleanly() -> None:
    """Verifica el cambio entre casos demostrativos y la superposición de conmutatividad."""
    at = AppTest.from_file(APP_PATH, default_timeout=60)
    at.run()
    assert not at.exception

    options = list(at.sidebar.selectbox[0].options)
    for option in options[1:]:
        at.sidebar.selectbox[0].select(option).run()
        assert not at.exception

    # Activar casilla de superposición de orden invertido (conmutatividad)
    assert len(at.checkbox) == 1
    at.checkbox[0].check().run()
    assert not at.exception
    assert len(at.success) == 1
