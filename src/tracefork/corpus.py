"""Corpus-wide blame/Shapley aggregation and z-score regression detection.

`store.py`'s `causal_edges` table already persists every `blame.py`
`FlipRateResult`/`ShapleyResult` a caller chose to keep
(`save_blame_report`/`save_shapley_report`), keyed by run — but nothing reads
it back *across* runs. This module is the read-only aggregation layer over
that existing data, built purely on `TapeStore.list_runs()` (run_id/
agent_name/created_at) and `TapeStore.causal_edges_for_run()` (the persisted
edge rows): a corpus-wide "who's most responsible, across every run we've
ever blamed" index, plus a simple z-score outlier check over each
`(agent_name, step_index, method)`'s history that flags a step whose most
recent flip_rate/shapley_value jumped well outside its own prior spread —
the "did this agent's causal profile just change" regression-dashboard slice
(a full web/report.html panel is out of scope here; see `tracefork
corpus-blame`'s module-level CLI docstring for the deferral rationale).

Both surfaces are read-only and add zero new SQL: they only call
`TapeStore.list_runs()`/`causal_edges_for_run()`, never write to the store,
and never touch `Tape.digest()`/`to_bytes()`/`from_bytes()` — this is pure
aggregation over data the graph store already has.

`CorpusEdgeSummary.created_at` is the edge's RUN's `created_at` (from
`list_runs()`), not the `causal_edges` row's own `created_at` (which
reflects when that edge was last (re-)blamed, not when the tape was
recorded) — the same lexical-ISO-string ordering convention
`TapeStore.prune()`'s `older_than_iso` cutoff already relies on, so grouping
a step's history "in run order" is well-defined without parsing dates.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .store import TapeStore


@dataclass(frozen=True)
class CorpusEdgeSummary:
    """One ``causal_edges`` row joined with its run's ``agent_name``/
    ``created_at`` (via ``TapeStore.list_runs()``) — the per-edge unit
    :func:`build_corpus_blame_index`/:func:`detect_regressions` both operate
    on. ``created_at`` is the RUN's, not the edge's own (see module
    docstring)."""

    edge_id: str
    run_id: str
    agent_name: str
    created_at: str
    step_index: int
    method: str
    flip_rate: float | None
    ci_lo: float | None
    ci_hi: float | None
    p_value: float | None
    q_value: float | None
    responsible: bool | None
    necessity: bool | None
    sufficiency: bool | None
    shapley_value: float | None

    @property
    def score(self) -> float:
        """The scalar both aggregations rank/compare by: ``flip_rate`` for a
        ``method="blame"`` edge, ``shapley_value`` for a ``method="shapley"``
        edge — each method's own "how responsible" number, on its own
        native scale (never averaged or normalized across methods). ``0.0``
        if the relevant field is unset (defensive only; every edge
        `save_blame_report`/`save_shapley_report` writes sets it)."""
        value = self.flip_rate if self.method == "blame" else self.shapley_value
        return value if value is not None else 0.0


@dataclass(frozen=True)
class CorpusBlameIndex:
    """Corpus-wide summary returned by :func:`build_corpus_blame_index`."""

    run_count: int
    edge_count: int
    by_method: dict[str, int]
    top_responsible: list[CorpusEdgeSummary]


@dataclass(frozen=True)
class RegressionFlag:
    """One ``(agent_name, step_index, method)`` whose latest run's score
    cleared :func:`detect_regressions`'s z-score threshold against that
    same step's own prior history."""

    agent_name: str
    step_index: int
    method: str
    run_id: str
    created_at: str
    value: float
    history_mean: float
    history_stdev: float
    z_score: float


def _corpus_edge_summaries(store: TapeStore) -> list[CorpusEdgeSummary]:
    """Every ``causal_edges`` row in ``store``, joined with its run's
    ``agent_name``/``created_at``. Shared helper for both aggregations below
    so there is exactly one join over ``list_runs()``/``causal_edges_for_run()``."""
    summaries: list[CorpusEdgeSummary] = []
    for run in store.list_runs():
        for edge in store.causal_edges_for_run(run["run_id"]):
            summaries.append(
                CorpusEdgeSummary(
                    edge_id=edge["edge_id"],
                    run_id=edge["run_id"],
                    agent_name=run["agent_name"],
                    created_at=run["created_at"],
                    step_index=edge["step_index"],
                    method=edge["method"],
                    flip_rate=edge["flip_rate"],
                    ci_lo=edge["ci_lo"],
                    ci_hi=edge["ci_hi"],
                    p_value=edge["p_value"],
                    q_value=edge["q_value"],
                    responsible=edge["responsible"],
                    necessity=edge["necessity"],
                    sufficiency=edge["sufficiency"],
                    shapley_value=edge["shapley_value"],
                )
            )
    return summaries


def build_corpus_blame_index(store: TapeStore, *, top_n: int = 20) -> CorpusBlameIndex:
    """Corpus-wide blame/Shapley index over every run in ``store``.

    Loops every run (``TapeStore.list_runs()``) and every one of its
    persisted ``causal_edges`` rows (``causal_edges_for_run()``), joining
    each edge with its run's ``agent_name``/``created_at``. ``by_method`` is
    a count per ``method`` (``"blame"``/``"shapley"``) across the whole
    corpus; ``top_responsible`` is every joined edge sorted descending by
    :attr:`CorpusEdgeSummary.score` (blame and Shapley edges mixed together,
    each ranked on its own native scale), capped at ``top_n``. Read-only —
    issues no writes and adds no new SQL beyond the two existing
    ``TapeStore`` calls.
    """
    runs = store.list_runs()
    summaries = _corpus_edge_summaries(store)

    by_method: dict[str, int] = {}
    for summary in summaries:
        by_method[summary.method] = by_method.get(summary.method, 0) + 1

    top_responsible = sorted(summaries, key=lambda s: s.score, reverse=True)[:top_n]

    return CorpusBlameIndex(
        run_count=len(runs),
        edge_count=len(summaries),
        by_method=by_method,
        top_responsible=top_responsible,
    )


def detect_regressions(
    store: TapeStore,
    *,
    method: str = "blame",
    z_threshold: float = 2.0,
    min_history: int = 3,
) -> list[RegressionFlag]:
    """Flag ``(agent_name, step_index)`` pairs whose most recent run's score
    is a statistical outlier against that same pair's own prior history.

    Filters the corpus (see :func:`build_corpus_blame_index`'s join) to
    edges of the given ``method`` only (``flip_rate`` for ``"blame"``,
    ``shapley_value`` for ``"shapley"`` — see
    :attr:`CorpusEdgeSummary.score`), groups by ``(agent_name, step_index,
    method)``, and sorts each group by its RUN's ``created_at`` (lexical
    ISO, the same convention ``TapeStore.prune()``'s ``older_than_iso``
    already relies on — see module docstring). The last point in a sorted
    group is the "latest" observation; every point before it is that
    step's "history". A group with fewer than ``min_history`` HISTORY
    points (i.e. total group size ``<= min_history``) has too little
    signal to judge and is skipped entirely, as is a group whose history
    has zero variance (population stdev is 0 — nothing to compare an
    outlier against without a spurious divide-by-zero).

    Flags the latest point when ``abs(z_score) >= z_threshold``, where
    ``z_score = (latest.score - history_mean) / history_stdev`` (population
    mean/stdev via stdlib ``statistics``, computed over the history points
    only — the latest point is never folded into its own baseline).
    Returned :class:`RegressionFlag` rows are sorted by ``abs(z_score)``
    descending (the most anomalous step first). Read-only — never writes to
    ``store``.
    """
    groups: dict[tuple[str, int, str], list[CorpusEdgeSummary]] = {}
    for summary in _corpus_edge_summaries(store):
        if summary.method != method:
            continue
        key = (summary.agent_name, summary.step_index, summary.method)
        groups.setdefault(key, []).append(summary)

    flags: list[RegressionFlag] = []
    for (agent_name, step_index, method_name), points in groups.items():
        if len(points) <= min_history:
            continue
        points = sorted(points, key=lambda s: s.created_at)
        *history, latest = points

        history_scores = [p.score for p in history]
        history_mean = statistics.mean(history_scores)
        history_stdev = statistics.pstdev(history_scores)
        if history_stdev == 0.0:
            continue

        z_score = (latest.score - history_mean) / history_stdev
        if abs(z_score) >= z_threshold:
            flags.append(
                RegressionFlag(
                    agent_name=agent_name,
                    step_index=step_index,
                    method=method_name,
                    run_id=latest.run_id,
                    created_at=latest.created_at,
                    value=latest.score,
                    history_mean=history_mean,
                    history_stdev=history_stdev,
                    z_score=z_score,
                )
            )

    flags.sort(key=lambda f: abs(f.z_score), reverse=True)
    return flags
