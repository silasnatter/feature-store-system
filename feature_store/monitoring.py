"""Monitoring: drift, freshness, a log of every check, and the current status.

Drift is measured with the population stability index (PSI). It compares how
a feature's values are spread over a set of bins today with how they were
spread in the data the model was trained on:

    PSI = sum over bins of (share_now - share_then) * ln(share_now / share_then)

0 means the two distributions are identical. By convention, below 0.1 counts
as stable, 0.1 to 0.25 as a moderate shift and above 0.25 as a significant one.
"""

import math
import statistics
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from datetime import datetime

import psycopg

from feature_store.validation import ALERT, OK, WARNING, Check

PSI_WARNING = 0.1
PSI_ALERT = 0.25

# Statuses from harmless to serious, for picking the worst of several
_SEVERITY = {OK: 0, WARNING: 1, ALERT: 2}


def _is_missing(value: float | None) -> bool:
    return value is None or value != value  # NaN is the only value not equal to itself


def _shares(values: Sequence[float | None], edges: Sequence[float]) -> list[float]:
    """The share of values in each bin. The last entry is the share of missing values.

    The edges cut the number line into len(edges) + 1 bins; a value equal to an
    edge belongs to the bin below it.
    """
    counts = [0] * (len(edges) + 2)
    for value in values:
        if _is_missing(value):
            counts[-1] += 1
        else:
            counts[bisect_left(edges, value)] += 1
    return [count / len(values) for count in counts]


def reference_distribution(values: Iterable[float | None], bins: int = 10) -> dict:
    """Describe how values are distributed, as bin edges and the share per bin.

    The edges are the values' own quantiles, so each bin starts out holding
    about the same share. Features with many repeated values (mostly zeros,
    say) get fewer bins, because repeated edges are merged.
    """
    values = list(values)
    present = [value for value in values if not _is_missing(value)]
    edges = (
        sorted(set(statistics.quantiles(present, n=bins, method="inclusive")))
        if len(present) >= 2
        else []
    )
    return {"edges": edges, "shares": _shares(values, edges)}


def population_stability_index(reference: dict, values: Sequence[float | None]) -> float:
    """How far `values` have moved from the reference distribution; 0 is identical."""
    # A bin that is empty on one side would make the logarithm infinite
    floor = 1e-4
    psi = 0.0
    for then, now in zip(reference["shares"], _shares(values, reference["edges"]), strict=True):
        then, now = max(then, floor), max(now, floor)
        psi += (now - then) * math.log(now / then)
    return psi


def drift_checks(
    conn: psycopg.Connection, as_of: datetime, reference: dict[str, dict]
) -> list[Check]:
    """Compare each feature's values at `as_of` with its reference distribution.

    `reference` maps feature names to the output of `reference_distribution`.
    """
    checks = []
    for name, distribution in sorted(reference.items()):
        rows = conn.execute(
            """
            SELECT fval.value
            FROM feature_store.feature_values fval
            JOIN feature_store.feature_definitions fd USING (feature_id)
            WHERE fd.feature_name = %s
              AND fval.event_timestamp = %s
            """,
            (name, as_of),
        ).fetchall()
        if not rows:
            message = f"not checked: no values for {as_of.isoformat()}"
            checks.append(Check(name, "psi", None, WARNING, message))
            continue
        psi = population_stability_index(distribution, [row["value"] for row in rows])
        status = ALERT if psi >= PSI_ALERT else WARNING if psi >= PSI_WARNING else OK
        message = (
            f"PSI {psi:.3f} against the model's training data "
            f"(warning from {PSI_WARNING}, alert from {PSI_ALERT})"
        )
        checks.append(Check(name, "psi", psi, status, message))
    return checks


def freshness_checks(conn: psycopg.Connection, as_of: datetime) -> list[Check]:
    """For each active feature: how old are its newest values at `as_of`?"""
    features = conn.execute(
        """
        SELECT fd.feature_name, fd.freshness_hours, newest.event_timestamp
        FROM feature_store.feature_definitions fd
        LEFT JOIN LATERAL (
            SELECT fval.event_timestamp
            FROM feature_store.feature_values fval
            WHERE fval.feature_id = fd.feature_id
              AND fval.event_timestamp <= %(as_of)s
            ORDER BY fval.event_timestamp DESC
            LIMIT 1
        ) AS newest ON true
        WHERE fd.status = 'active'
        ORDER BY fd.feature_name
        """,
        {"as_of": as_of},
    ).fetchall()

    checks = []
    for feature in features:
        name, allowed = feature["feature_name"], feature["freshness_hours"]
        if feature["event_timestamp"] is None:
            checks.append(Check(name, "freshness_hours", None, ALERT, "no values at all"))
            continue
        age = (as_of - feature["event_timestamp"]).total_seconds() / 3600
        message = f"newest values are {age:.1f} hours old, allowed {allowed}"
        checks.append(Check(name, "freshness_hours", age, ALERT if age > allowed else OK, message))
    return checks


def log_checks(
    conn: psycopg.Connection, kind: str, snapshot_at: datetime, checks: Iterable[Check]
) -> None:
    """Record check outcomes; an earlier outcome for the same snapshot is replaced.

    `kind` is 'feature' or 'model'. Does not commit.
    """
    for check in checks:
        conn.execute(
            """
            INSERT INTO feature_store.monitor_logs
                (kind, name, metric, snapshot_at, value, status, message)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (kind, name, metric, snapshot_at) DO UPDATE SET
                value = EXCLUDED.value,
                status = EXCLUDED.status,
                message = EXCLUDED.message,
                checked_at = now()
            """,
            (kind, check.name, check.metric, snapshot_at, check.value, check.status, check.message),
        )


def worst(statuses: Iterable[str]) -> str:
    return max(statuses, key=_SEVERITY.__getitem__, default=OK)


def current_status(conn: psycopg.Connection, as_of: datetime) -> dict:
    """The health of every feature and model at `as_of`.

    Takes the most recent logged outcome of each check, up to `as_of`, and adds
    freshness, which is measured against `as_of` itself.
    """
    logged = conn.execute(
        """
        SELECT DISTINCT ON (kind, name, metric)
            kind, name, metric, value, status, message, snapshot_at
        FROM feature_store.monitor_logs
        WHERE snapshot_at <= %s
        ORDER BY kind, name, metric, snapshot_at DESC
        """,
        (as_of,),
    ).fetchall()

    def entry(check_row: dict) -> dict:
        return {key: check_row[key] for key in ("metric", "value", "status", "message")} | {
            "snapshot_at": check_row["snapshot_at"]
        }

    groups: dict[str, dict[str, list[dict]]] = {"feature": {}, "model": {}}
    for row in logged:
        groups[row["kind"]].setdefault(row["name"], []).append(entry(row))
    for check in freshness_checks(conn, as_of):
        groups["feature"].setdefault(check.name, []).append(
            {
                "metric": check.metric,
                "value": check.value,
                "status": check.status,
                "message": check.message,
                "snapshot_at": as_of,
            }
        )

    def summarise(by_name: dict[str, list[dict]]) -> dict:
        return {
            name: {"status": worst(check["status"] for check in checks), "checks": checks}
            for name, checks in sorted(by_name.items())
        }

    features, models = summarise(groups["feature"]), summarise(groups["model"])
    overall = worst(item["status"] for item in [*features.values(), *models.values()])
    return {"status": overall, "as_of": as_of, "features": features, "models": models}
