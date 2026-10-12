"""Pruebas de integridad del catálogo sintético incluido."""

from diagnosis_simulator.sample_data import get_sample_catalog


def test_sample_catalog_is_complete_and_valid() -> None:
    catalog = get_sample_catalog()

    catalog.validate()
    assert len(catalog.conditions) == 4
    assert len(catalog.symptoms) == 6
