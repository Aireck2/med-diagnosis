"""SQLite database builder and repository adapter for epidemiological parameters.

Seeds and queries a normalized 3NF SQLite database from `schema.sql` and
`epidemiology_seed.json`, verifying exact rational consistency with Table 1
of MateDiscreta.pdf.
"""

from __future__ import annotations

import json
import sqlite3
from fractions import Fraction
from pathlib import Path

from med_diagnosis.domain.models import (
    Citation,
    DiagnosticTestSpec,
    ParameterMatrixRow,
    PathologyProfile,
    PrevalenceScenario,
)

_DATA_DIR = Path(__file__).resolve().parent
DEFAULT_SCHEMA_PATH = _DATA_DIR / "schema.sql"
DEFAULT_SEED_PATH = _DATA_DIR / "epidemiology_seed.json"
DEFAULT_DB_PATH = _DATA_DIR / "epidemiology.db"


def _populate_connection(
    conn: sqlite3.Connection,
    schema_sql: str,
    seed_payload: dict[str, object],
) -> None:
    """Apply DDL schema and insert validated seed records into a SQLite connection."""
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(schema_sql)
    conn.execute("PRAGMA foreign_keys = ON;")

    with conn:
        conn.execute("DELETE FROM pathology_citations;")
        conn.execute("DELETE FROM diagnostic_tests;")
        conn.execute("DELETE FROM prevalence_scenarios;")
        conn.execute("DELETE FROM pathologies;")
        conn.execute("DELETE FROM citations;")

        citations = seed_payload["citations"]
        assert isinstance(citations, list)
        for cite in citations:
            conn.execute(
                """
                INSERT INTO citations (cite_key, short_label, full_reference, pub_year, doi_or_url)
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    cite["cite_key"],
                    cite["short_label"],
                    cite["full_reference"],
                    int(cite["pub_year"]),
                    cite.get("doi_or_url", ""),
                ),
            )

        pathologies = seed_payload["pathologies"]
        assert isinstance(pathologies, list)
        for pathology in pathologies:
            slug = str(pathology["slug"])
            conn.execute(
                """
                INSERT INTO pathologies
                (slug, name, target_population, clinical_context, display_order)
                VALUES (?, ?, ?, ?, ?);
                """,
                (
                    slug,
                    pathology["name"],
                    pathology["target_population"],
                    pathology["clinical_context"],
                    int(pathology["display_order"]),
                ),
            )

            for idx, cite_key in enumerate(pathology["citations"], start=1):
                conn.execute(
                    """
                    INSERT INTO pathology_citations (pathology_slug, cite_key, citation_order)
                    VALUES (?, ?, ?);
                    """,
                    (slug, cite_key, idx),
                )

            for scenario in pathology["scenarios"]:
                conn.execute(
                    """
                    INSERT INTO prevalence_scenarios
                    (pathology_slug, stratum_name, clinical_description,
                     prior_num, prior_den, display_order)
                    VALUES (?, ?, ?, ?, ?, ?);
                    """,
                    (
                        slug,
                        scenario["stratum_name"],
                        scenario["clinical_description"],
                        int(scenario["prior_num"]),
                        int(scenario["prior_den"]),
                        int(scenario["display_order"]),
                    ),
                )

            for test in pathology["tests"]:
                sens = Fraction(int(test["sensitivity_num"]), int(test["sensitivity_den"]))
                spec = Fraction(int(test["specificity_num"]), int(test["specificity_den"]))
                expected_fp = Fraction(int(test["expected_fp_num"]), int(test["expected_fp_den"]))
                if Fraction(1) - spec != expected_fp:
                    raise ValueError(
                        f"Table 1 invariant violated for {test['test_name']}: "
                        f"1 - {spec} != {expected_fp}"
                    )
                if not (Fraction(0) < sens <= Fraction(1)):
                    raise ValueError(f"Invalid sensitivity for {test['test_name']}: {sens}")

                conn.execute(
                    """
                    INSERT INTO diagnostic_tests
                    (pathology_slug, default_step_order, test_name,
                     sensitivity_num, sensitivity_den, specificity_num, specificity_den,
                     biological_mechanism)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        slug,
                        int(test["default_step_order"]),
                        test["test_name"],
                        int(test["sensitivity_num"]),
                        int(test["sensitivity_den"]),
                        int(test["specificity_num"]),
                        int(test["specificity_den"]),
                        test["biological_mechanism"],
                    ),
                )


def build_epidemiology_database(
    db_path: Path = DEFAULT_DB_PATH,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    seed_path: Path = DEFAULT_SEED_PATH,
) -> Path:
    """Create or rebuild the SQLite epidemiological database file on disk."""
    schema_sql = schema_path.read_text(encoding="utf-8")
    seed_payload = json.loads(seed_path.read_text(encoding="utf-8"))
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        _populate_connection(conn, schema_sql, seed_payload)
    finally:
        conn.close()
    return db_path


class SqliteEpidemiologyRepository:
    """Read-only repository querying the epidemiological SQLite database."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path
        self._memory_conn: sqlite3.Connection | None = None
        if db_path is None or not db_path.exists():
            # Fallback to an in-memory SQLite instance seeded deterministically
            # so read-only cloud deployments never fail even if .db is missing.
            schema_sql = DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8")
            seed_payload = json.loads(DEFAULT_SEED_PATH.read_text(encoding="utf-8"))
            conn = sqlite3.connect(":memory:", check_same_thread=False)
            conn.row_factory = sqlite3.Row
            _populate_connection(conn, schema_sql, seed_payload)
            self._memory_conn = conn

    def _connect(self) -> tuple[sqlite3.Connection, bool]:
        """Return (connection, should_close)."""
        if self._memory_conn is not None:
            return self._memory_conn, False
        assert self._db_path is not None
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn, True

    def list_pathologies(self) -> tuple[PathologyProfile, ...]:
        """Return all pathologies ordered by display_order."""
        conn, should_close = self._connect()
        try:
            rows = conn.execute(
                "SELECT slug FROM pathologies ORDER BY display_order ASC;"
            ).fetchall()
            return tuple(self.get_pathology(str(row["slug"])) for row in rows)
        finally:
            if should_close:
                conn.close()

    def get_pathology(self, slug: str) -> PathologyProfile:
        """Load complete PathologyProfile with scenarios, tests, and citations."""
        conn, should_close = self._connect()
        try:
            p_row = conn.execute(
                """
                SELECT slug, name, target_population, clinical_context
                FROM pathologies
                WHERE slug = ?;
                """,
                (slug,),
            ).fetchone()
            if p_row is None:
                raise KeyError(f"Unknown pathology slug: {slug!r}")

            s_rows = conn.execute(
                """
                SELECT id, pathology_slug, stratum_name, clinical_description,
                       prior_num, prior_den, display_order
                FROM prevalence_scenarios
                WHERE pathology_slug = ?
                ORDER BY display_order ASC;
                """,
                (slug,),
            ).fetchall()
            scenarios = tuple(
                PrevalenceScenario(
                    scenario_id=int(r["id"]),
                    pathology_slug=str(r["pathology_slug"]),
                    stratum_name=str(r["stratum_name"]),
                    clinical_description=str(r["clinical_description"]),
                    prior=Fraction(int(r["prior_num"]), int(r["prior_den"])),
                    display_order=int(r["display_order"]),
                )
                for r in s_rows
            )

            t_rows = conn.execute(
                """
                SELECT id, pathology_slug, default_step_order, test_name,
                       sensitivity_num, sensitivity_den, specificity_num, specificity_den,
                       biological_mechanism
                FROM diagnostic_tests
                WHERE pathology_slug = ?
                ORDER BY default_step_order ASC;
                """,
                (slug,),
            ).fetchall()
            tests = tuple(
                DiagnosticTestSpec(
                    test_id=int(r["id"]),
                    pathology_slug=str(r["pathology_slug"]),
                    default_step_order=int(r["default_step_order"]),
                    name=str(r["test_name"]),
                    sensitivity=Fraction(int(r["sensitivity_num"]), int(r["sensitivity_den"])),
                    specificity=Fraction(int(r["specificity_num"]), int(r["specificity_den"])),
                    biological_mechanism=str(r["biological_mechanism"]),
                )
                for r in t_rows
            )

            c_rows = conn.execute(
                """
                SELECT c.cite_key, c.short_label, c.full_reference, c.pub_year, c.doi_or_url
                FROM citations c
                JOIN pathology_citations pc ON pc.cite_key = c.cite_key
                WHERE pc.pathology_slug = ?
                ORDER BY pc.citation_order ASC;
                """,
                (slug,),
            ).fetchall()
            citations = tuple(
                Citation(
                    cite_key=str(r["cite_key"]),
                    short_label=str(r["short_label"]),
                    full_reference=str(r["full_reference"]),
                    pub_year=int(r["pub_year"]),
                    doi_or_url=str(r["doi_or_url"]),
                )
                for r in c_rows
            )

            return PathologyProfile(
                slug=str(p_row["slug"]),
                name=str(p_row["name"]),
                target_population=str(p_row["target_population"]),
                clinical_context=str(p_row["clinical_context"]),
                scenarios=scenarios,
                tests=tests,
                citations=citations,
            )
        finally:
            if should_close:
                conn.close()

    def list_all_citations(self) -> tuple[Citation, ...]:
        """Return all bibliographic references ordered alphabetically by short_label."""
        conn, should_close = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT cite_key, short_label, full_reference, pub_year, doi_or_url
                FROM citations
                ORDER BY short_label ASC;
                """
            ).fetchall()
            return tuple(
                Citation(
                    cite_key=str(r["cite_key"]),
                    short_label=str(r["short_label"]),
                    full_reference=str(r["full_reference"]),
                    pub_year=int(r["pub_year"]),
                    doi_or_url=str(r["doi_or_url"]),
                )
                for r in rows
            )
        finally:
            if should_close:
                conn.close()

    def list_parameter_matrix(self) -> tuple[ParameterMatrixRow, ...]:
        """Return flat parameter matrix matching Table 1 of MateDiscreta.pdf."""
        rows: list[ParameterMatrixRow] = []
        for profile in self.list_pathologies():
            priors_summary = " · ".join(
                f"{s.stratum_name}: {float(s.prior) * 100:.2f}".rstrip("0").rstrip(".") + "%"
                for s in profile.scenarios
            )
            citations_summary = "; ".join(c.short_label for c in profile.citations)
            for test in profile.tests:
                rows.append(
                    ParameterMatrixRow(
                        pathology_slug=profile.slug,
                        pathology_name=profile.name,
                        target_population=profile.target_population,
                        default_step_order=test.default_step_order,
                        test_name=test.name,
                        sensitivity=test.sensitivity,
                        specificity=test.specificity,
                        false_positive_rate=test.false_positive_rate,
                        biological_mechanism=test.biological_mechanism,
                        priors_summary=priors_summary,
                        citations_summary=citations_summary,
                    )
                )
        return tuple(rows)


def load_default_repository() -> SqliteEpidemiologyRepository:
    """Return a repository backed by the packaged SQLite database (or seeded memory fallback)."""
    if DEFAULT_DB_PATH.exists():
        return SqliteEpidemiologyRepository(DEFAULT_DB_PATH)
    return SqliteEpidemiologyRepository(None)
