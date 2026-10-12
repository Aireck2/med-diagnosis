"""Punto de entrada de la aplicación Streamlit."""

import streamlit as st

from diagnosis_simulator.presentation import render_simulator
from diagnosis_simulator.sample_data import get_sample_catalog

st.set_page_config(
    page_title="Simulador bayesiano educativo",
    page_icon="🩺",
    layout="wide",
)

render_simulator(get_sample_catalog())
