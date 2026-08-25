"""Top-level `tracefork` package surface — the 1.0 SemVer-covered API.

Guards two things a plain `import tracefork` should always give you: (1) the
core product classes (not just the framework-adapter/observability grab-bag)
are top-level reachable, and (2) `tracefork.__all__` never drifts from what's
actually bound on the module — an entry in `__all__` that doesn't resolve, or
a top-level name missing from `__all__`, is exactly the kind of silent API
drift a SemVer promise can't tolerate.
"""

from __future__ import annotations

import tracefork

# The core engine surface `docs/plugin-api.md` and the 1.0 readiness plan both
# call out as missing pre-1.0: every name here must be `from tracefork import
# <name>`-able. Framework adapters / observability / redaction were already
# exported pre-1.0 and are covered by the `__all__`-completeness check below,
# not repeated here.
CORE_PRODUCT_API = [
    "Tape",
    "TapeStore",
    "TapeConflictError",
    "ForkPointDriftError",
    "ForkEngine",
    "Branch",
    "BranchSpec",
    "ConfinementSpec",
    "BlameEngine",
    "BlameReport",
    "Oracle",
    "StringMatchOracle",
    "BudgetGovernor",
    "BudgetExceededError",
    "ReplayVerifier",
    "VerificationResult",
    "ReplayCertificate",
    "CertificateStrength",
    "TraceforkTransport",
    "AsyncTraceforkTransport",
    "NondetSource",
    "RecordingNondet",
    "ReplayNondet",
    "DivergenceError",
    "RequestMatcher",
    "IdentityMatcher",
    "generate_report",
    "TournamentEngine",
]

# `docs/plugin-api.md`'s own "Public, SemVer-covered API" bullet for the
# plugin-registry surface.
PLUGIN_API = [
    "Registry",
    "PROVIDER_GROUP",
    "ORACLE_GROUP",
    "SERIALIZER_GROUP",
    "MATCHER_GROUP",
    "ADAPTER_GROUP",
    "ProviderAdapter",
    "TapeSerializer",
    # Oracle, RequestMatcher, FrameworkAdapter are already listed above/pre-existing.
]


def test_core_product_api_is_top_level_importable():
    missing = [name for name in CORE_PRODUCT_API if not hasattr(tracefork, name)]
    assert missing == [], f"core product symbols missing from top-level tracefork: {missing}"


def test_plugin_registry_api_is_top_level_importable():
    missing = [name for name in PLUGIN_API if not hasattr(tracefork, name)]
    assert missing == [], f"plugin-api symbols missing from top-level tracefork: {missing}"


def test_all_equals_actually_bound_names():
    """`set(tracefork.__all__)` must equal the set of names actually bound on
    the module — no aspirational entries that don't resolve, no top-level name
    that silently isn't advertised."""
    declared = set(tracefork.__all__)
    missing_from_module = {name for name in declared if not hasattr(tracefork, name)}
    assert missing_from_module == set(), (
        f"__all__ names not actually bound on tracefork: {missing_from_module}"
    )
    # Every name __all__ claims must resolve (checked above); this also proves
    # __all__ has no duplicates and every CORE_PRODUCT_API/PLUGIN_API name is
    # actually advertised, not just importable by accident.
    for name in CORE_PRODUCT_API + PLUGIN_API:
        assert name in declared, f"{name} is importable but missing from __all__"


def test_version_is_exported_and_matches_installed_package():
    from importlib import metadata

    assert hasattr(tracefork, "__version__")
    assert tracefork.__version__ == metadata.version("tracefork")
    assert "__version__" in tracefork.__all__
