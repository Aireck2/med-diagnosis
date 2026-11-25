-- Esquema relacional en Tercera Forma Normal (3FN) para la base de datos epidemiológica
-- Fuente de especificación: MateDiscreta.pdf (Sección 1.2 y Tabla 1)

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS pathologies (
    slug TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    target_population TEXT NOT NULL,
    clinical_context TEXT NOT NULL,
    display_order INTEGER NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS prevalence_scenarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pathology_slug TEXT NOT NULL,
    stratum_name TEXT NOT NULL,
    clinical_description TEXT NOT NULL,
    prior_num INTEGER NOT NULL,
    prior_den INTEGER NOT NULL,
    display_order INTEGER NOT NULL,
    FOREIGN KEY (pathology_slug) REFERENCES pathologies(slug) ON DELETE CASCADE,
    UNIQUE (pathology_slug, stratum_name),
    UNIQUE (pathology_slug, display_order),
    CHECK (prior_den > 0),
    CHECK (prior_num > 0 AND prior_num < prior_den)
);

CREATE TABLE IF NOT EXISTS diagnostic_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pathology_slug TEXT NOT NULL,
    default_step_order INTEGER NOT NULL,
    test_name TEXT NOT NULL,
    sensitivity_num INTEGER NOT NULL,
    sensitivity_den INTEGER NOT NULL,
    specificity_num INTEGER NOT NULL,
    specificity_den INTEGER NOT NULL,
    biological_mechanism TEXT NOT NULL,
    FOREIGN KEY (pathology_slug) REFERENCES pathologies(slug) ON DELETE CASCADE,
    UNIQUE (pathology_slug, default_step_order),
    UNIQUE (pathology_slug, test_name),
    CHECK (default_step_order >= 1),
    CHECK (sensitivity_den > 0),
    CHECK (sensitivity_num > 0 AND sensitivity_num <= sensitivity_den),
    CHECK (specificity_den > 0),
    CHECK (specificity_num > 0 AND specificity_num <= specificity_den)
);

CREATE TABLE IF NOT EXISTS citations (
    cite_key TEXT PRIMARY KEY,
    short_label TEXT NOT NULL UNIQUE,
    full_reference TEXT NOT NULL,
    pub_year INTEGER NOT NULL,
    doi_or_url TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS pathology_citations (
    pathology_slug TEXT NOT NULL,
    cite_key TEXT NOT NULL,
    citation_order INTEGER NOT NULL,
    PRIMARY KEY (pathology_slug, cite_key),
    FOREIGN KEY (pathology_slug) REFERENCES pathologies(slug) ON DELETE CASCADE,
    FOREIGN KEY (cite_key) REFERENCES citations(cite_key) ON DELETE CASCADE
);
