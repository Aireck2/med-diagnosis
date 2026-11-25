"""AST boundary tests verifying the domain layer has zero external or UI dependencies."""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_TOP_LEVEL_IMPORTS = {"streamlit", "plotly", "sqlite3", "pandas", "numpy", "altair"}


def test_domain_layer_only_imports_stdlib_and_domain() -> None:
    domain_dir = Path(__file__).resolve().parents[1] / "src" / "med_diagnosis" / "domain"
    py_files = sorted(domain_dir.glob("*.py"))
    assert py_files, "Expected Python modules in med_diagnosis.domain"

    for py_file in py_files:
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top not in FORBIDDEN_TOP_LEVEL_IMPORTS, (
                        f"Forbidden import {alias.name!r} in {py_file.name}"
                    )
                    assert not alias.name.startswith("med_diagnosis.ui"), (
                        f"Domain must not import UI layer in {py_file.name}"
                    )
                    assert not alias.name.startswith("med_diagnosis.data"), (
                        f"Domain must not import data layer in {py_file.name}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                assert top not in FORBIDDEN_TOP_LEVEL_IMPORTS, (
                    f"Forbidden from-import {node.module!r} in {py_file.name}"
                )
                assert not node.module.startswith("med_diagnosis.ui"), (
                    f"Domain must not import UI layer in {py_file.name}"
                )
                assert not node.module.startswith("med_diagnosis.data"), (
                    f"Domain must not import data layer in {py_file.name}"
                )
