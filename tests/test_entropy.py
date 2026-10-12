"""Pruebas analíticas para la entropía de Shannon."""

from math import isclose

import pytest

from diagnosis_simulator.entropy import shannon_entropy


def test_uniform_distribution_has_maximum_entropy() -> None:
    assert isclose(shannon_entropy({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}), 2.0)


def test_deterministic_distribution_has_zero_entropy() -> None:
    assert shannon_entropy({"a": 1.0, "b": 0.0}) == 0.0


def test_zero_probability_contributes_zero() -> None:
    assert isclose(shannon_entropy({"a": 0.5, "b": 0.5, "c": 0.0}), 1.0)


@pytest.mark.parametrize(
    "distribution",
    [
        {},
        {"a": 0.4, "b": 0.4},
        {"a": -0.1, "b": 1.1},
        {"a": float("nan")},
        {"a": True},
    ],
)
def test_invalid_distribution_is_rejected(distribution: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        shannon_entropy(distribution)
