"""Tests for the SQLite epidemiological database and repository (MateDiscreta.pdf Table 1)."""

from __future__ import annotations

import sqlite3
from fractions import Fraction
from pathlib import Path

import pytest

from med_diagnosis.data.database import (
    SqliteEpidemiologyRepository,
    build_epidemiology_database,
    load_default_repository,
)


def test_default_repository_loads_all_three_pathologies() -> None:
    repo = load_default_repository()
    pathologies = repo.list_pathologies()

    assert [p.slug for p in pathologies] == ["covid19", "tb", "dengue"]
    assert [p.name for p in pathologies] == [
        "COVID-19 (SARS-CoV-2)",
        "Tuberculosis Pulmonar (TB)",
        "Dengue (DENV 1–4)",
    ]


def test_covid19_parameters_match_table_1_exact_fractions() -> None:
    repo = load_default_repository()
    covid = repo.get_pathology("covid19")

    assert [s.stratum_name for s in covid.scenarios] == [
        "Asintomático",
        "Sintomático IRA",
        "Contacto estrecho",
    ]
    assert [s.prior for s in covid.scenarios] == [
        Fraction(15, 1000),
        Fraction(120, 1000),
        Fraction(350, 1000),
    ]
    for scenario in covid.scenarios:
        assert scenario.prior + scenario.complement == Fraction(1, 1)

    assert [t.name for t in covid.tests] == [
        "Prueba Rápida de Antígenos",
        "Prueba Serológica (IgM/IgG)",
        "RT-PCR en Tiempo Real",
    ]
    assert [t.sensitivity for t in covid.tests] == [
        Fraction(730, 1000),
        Fraction(680, 1000),
        Fraction(950, 1000),
    ]
    assert [t.specificity for t in covid.tests] == [
        Fraction(991, 1000),
        Fraction(965, 1000),
        Fraction(995, 1000),
    ]
    assert [t.false_positive_rate for t in covid.tests] == [
        Fraction(9, 1000),
        Fraction(35, 1000),
        Fraction(5, 1000),
    ]
    assert [c.short_label for c in covid.citations] == [
        "CDC Perú (2024a)",
        "Dinnes et al. (2022)",
        "OMS (2021)",
    ]


def test_tb_parameters_match_table_1_exact_fractions() -> None:
    repo = load_default_repository()
    tb = repo.get_pathology("tb")

    assert [s.stratum_name for s in tb.scenarios] == [
        "Población general",
        "Sintomático Resp.",
        "Contacto TB activo",
    ]
    assert [s.prior for s in tb.scenarios] == [
        Fraction(15, 10000),
        Fraction(45, 1000),
        Fraction(150, 1000),
    ]

    assert [t.name for t in tb.tests] == [
        "Radiografía de Tórax",
        "Baciloscopía Esputo (AFB)",
        "GeneXpert MTB/RIF",
    ]
    assert [t.sensitivity for t in tb.tests] == [
        Fraction(870, 1000),
        Fraction(650, 1000),
        Fraction(890, 1000),
    ]
    assert [t.specificity for t in tb.tests] == [
        Fraction(890, 1000),
        Fraction(980, 1000),
        Fraction(990, 1000),
    ]
    assert [t.false_positive_rate for t in tb.tests] == [
        Fraction(110, 1000),
        Fraction(20, 1000),
        Fraction(10, 1000),
    ]
    assert [c.short_label for c in tb.citations] == [
        "MINSA (2023)",
        "OMS (2024)",
    ]


def test_dengue_parameters_match_table_1_exact_fractions() -> None:
    repo = load_default_repository()
    dengue = repo.get_pathology("dengue")

    assert [s.stratum_name for s in dengue.scenarios] == [
        "Fuera de brote",
        "Brote estacional",
    ]
    assert [s.prior for s in dengue.scenarios] == [
        Fraction(30, 1000),
        Fraction(220, 1000),
    ]

    assert [t.name for t in dengue.tests] == [
        "Prueba Rápida Antígeno NS1",
        "ELISA IgM Dengue",
        "RT-PCR Multiplex DENV",
    ]
    assert [t.sensitivity for t in dengue.tests] == [
        Fraction(760, 1000),
        Fraction(850, 1000),
        Fraction(940, 1000),
    ]
    assert [t.specificity for t in dengue.tests] == [
        Fraction(980, 1000),
        Fraction(920, 1000),
        Fraction(995, 1000),
    ]
    assert [t.false_positive_rate for t in dengue.tests] == [
        Fraction(20, 1000),
        Fraction(80, 1000),
        Fraction(5, 1000),
    ]
    assert [c.short_label for c in dengue.citations] == [
        "CDC Perú (2024b)",
        "OPS (2022)",
    ]


def test_database_totals_and_relational_constraints(tmp_path: Path) -> None:
    db_file = tmp_path / "test_epidemiology.db"
    build_epidemiology_database(db_file)
    repo = SqliteEpidemiologyRepository(db_file)

    assert len(repo.list_pathologies()) == 3
    assert len(repo.list_all_citations()) == 7
    matrix_rows = repo.list_parameter_matrix()
    assert len(matrix_rows) == 9

    with pytest.raises(KeyError, match="Unknown pathology"):
        repo.get_pathology("unknown-disease")

    conn = sqlite3.connect(db_file)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO prevalence_scenarios
                (pathology_slug, stratum_name, clinical_description,
                 prior_num, prior_den, display_order)
                VALUES ('covid19', 'Invalid Prior', 'Out of range', 1200, 1000, 99);
                """
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO diagnostic_tests
                (pathology_slug, default_step_order, test_name, sensitivity_num, sensitivity_den,
                 specificity_num, specificity_den, biological_mechanism)
                VALUES ('nonexistent', 1, 'Ghost Test', 90, 100, 90, 100, 'None');
                """
            )
    finally:
        conn.close()
