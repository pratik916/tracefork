"""Static metadata regression tests over pyproject.toml's [project] table.

Distinct from test_pyproject_packaging.py (which builds real wheel/sdist
artifacts and inspects their contents): these tests parse pyproject.toml
directly with stdlib tomllib -- no build, no subprocess -- because the
things under test here are metadata FIELDS, not build-artifact contents.

Locks in the 1.0.0 graduation metadata pass: version, Trove classifier
lifecycle status, and the PEP 639 `license`-as-SPDX-expression form (which
must not coexist with a deprecated `License :: ...` Trove classifier --
having both is exactly the ambiguity PEP 639 exists to remove).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


def _load_project_table() -> dict:
    with PYPROJECT_PATH.open("rb") as f:
        data = tomllib.load(f)
    return data["project"]


def test_version_is_1_0_0():
    project = _load_project_table()
    assert project["version"] == "1.0.0"


def test_development_status_is_production_stable_not_beta():
    classifiers = _load_project_table()["classifiers"]
    assert "Development Status :: 5 - Production/Stable" in classifiers
    assert "Development Status :: 4 - Beta" not in classifiers
    # Only one Development Status classifier should ever be present.
    status_classifiers = [c for c in classifiers if c.startswith("Development Status ::")]
    assert status_classifiers == ["Development Status :: 5 - Production/Stable"]


def test_license_is_pep639_spdx_expression_not_a_table():
    project = _load_project_table()
    # PEP 639: `license` is a bare SPDX expression string, e.g. "MIT" --
    # NOT the legacy `{text = "MIT"}` / `{file = "LICENSE"}` table form.
    assert isinstance(project["license"], str)
    assert project["license"] == "MIT"


def test_no_deprecated_license_classifier_alongside_spdx_expression():
    """PEP 639's whole point: a `License ::` Trove classifier is deprecated
    once `license` is a bare SPDX expression -- shipping both is the exact
    ambiguity (two disagreeing sources of truth for the same fact) PEP 639
    exists to remove. Guard against it regressing back in.
    """
    project = _load_project_table()
    assert isinstance(project["license"], str), "license must be a PEP 639 SPDX expression"
    classifiers = _load_project_table()["classifiers"]
    license_classifiers = [c for c in classifiers if c.startswith("License ::")]
    assert license_classifiers == [], (
        f"deprecated License :: classifier(s) found alongside license = "
        f"{project['license']!r}: {license_classifiers}"
    )


def test_project_urls_present_for_release():
    urls = _load_project_table()["urls"]
    for key in ("Homepage", "Repository", "Issues", "Changelog"):
        assert key in urls, f"project.urls missing {key!r}"
        assert urls[key].startswith("https://"), f"project.urls[{key!r}] is not a real URL"


def test_description_and_keywords_are_non_empty():
    project = _load_project_table()
    assert project["description"].strip()
    assert len(project["keywords"]) > 0
    assert all(isinstance(k, str) and k.strip() for k in project["keywords"])
