"""Quality checks on a computed snapshot.

Used twice: as a gate before values are copied to the online store
(`validate_snapshot`), and by the monitor, which records every outcome.
"""

from dataclasses import dataclass
from datetime import datetime

import psycopg

from feature_store.registry import FeatureRegistry

OK = "ok"
WARNING = "warning"
ALERT = "alert"


@dataclass(frozen=True)
class Check:
    """The outcome of one check."""

    name: str  # what was checked: a feature name or a model name
    metric: str
    value: float | None
    status: str  # OK, WARNING or ALERT
    message: str


def quality_checks(conn: psycopg.Connection, view_name: str, as_of: datetime) -> list[Check]:
    """Check the values a view has for one moment against its feature definitions.

    Per feature:
      - `row_count`: values must exist for that moment
      - `null_share`: the share of nulls must not exceed `null_threshold`
      - `minimum` and `maximum`: values must lie within `expected_min` and
        `expected_max`, where those are set
    """
    FeatureRegistry(conn).get_view(view_name)  # raises if the view is unknown
    features = conn.execute(
        """
        SELECT
            fd.feature_name, fd.expected_min, fd.expected_max, fd.null_threshold,
            count(fval.feature_id) AS n_values,
            count(fval.feature_id) FILTER (WHERE fval.value IS NULL) AS n_null,
            min(fval.value) AS min_value,
            max(fval.value) AS max_value
        FROM feature_store.feature_definitions fd
        JOIN feature_store.feature_views fv USING (view_id)
        LEFT JOIN feature_store.feature_values fval
            ON fval.feature_id = fd.feature_id
           AND fval.event_timestamp = %(as_of)s
        WHERE fv.view_name = %(view_name)s
          AND fd.status <> 'deprecated'
        GROUP BY fd.feature_name, fd.expected_min, fd.expected_max, fd.null_threshold
        ORDER BY fd.feature_name
        """,
        {"view_name": view_name, "as_of": as_of},
    ).fetchall()

    if not features:
        return [Check(view_name, "feature_count", 0, ALERT, "the view has no features")]

    checks = []
    for feature in features:
        name = feature["feature_name"]
        n_values = feature["n_values"]
        if n_values == 0:
            message = f"no values for {as_of.isoformat()}"
            checks.append(Check(name, "row_count", 0, ALERT, message))
            continue
        checks.append(Check(name, "row_count", n_values, OK, f"{n_values:,} values"))

        null_share = feature["n_null"] / n_values
        checks.append(
            Check(
                name,
                "null_share",
                null_share,
                ALERT if null_share > feature["null_threshold"] else OK,
                f"{null_share:.1%} of values are null, allowed {feature['null_threshold']:.1%}",
            )
        )

        # min and max are None when every value is null
        low, expected_low = feature["min_value"], feature["expected_min"]
        if expected_low is not None and low is not None:
            if low < expected_low:
                check = Check(
                    name,
                    "minimum",
                    low,
                    ALERT,
                    f"minimum {low:g} is below the expected {expected_low:g}",
                )
            else:
                check = Check(
                    name, "minimum", low, OK, f"minimum {low:g}, expected at least {expected_low:g}"
                )
            checks.append(check)

        high, expected_high = feature["max_value"], feature["expected_max"]
        if expected_high is not None and high is not None:
            if high > expected_high:
                check = Check(
                    name,
                    "maximum",
                    high,
                    ALERT,
                    f"maximum {high:g} is above the expected {expected_high:g}",
                )
            else:
                check = Check(
                    name,
                    "maximum",
                    high,
                    OK,
                    f"maximum {high:g}, expected at most {expected_high:g}",
                )
            checks.append(check)
    return checks


def validate_snapshot(conn: psycopg.Connection, view_name: str, as_of: datetime) -> list[str]:
    """Problems with the values a view has for one moment; an empty list means none."""
    return [
        f"{check.name}: {check.message}"
        for check in quality_checks(conn, view_name, as_of)
        if check.status == ALERT
    ]
