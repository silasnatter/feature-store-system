"""Checks on a computed snapshot, run before it is copied to the online store."""

from datetime import datetime

import psycopg

from feature_store.registry import FeatureRegistry


def validate_snapshot(conn: psycopg.Connection, view_name: str, as_of: datetime) -> list[str]:
    """Problems with the values a view has for one moment; an empty list means none.

    Per feature, against the thresholds in its definition:
      - values must exist for that moment
      - the share of nulls must not exceed `null_threshold`
      - values must lie within `expected_min` and `expected_max`, where set
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
        return [f"{view_name}: the view has no features"]

    problems = []
    for feature in features:
        name = feature["feature_name"]
        if feature["n_values"] == 0:
            problems.append(f"{name}: no values for {as_of.isoformat()}")
            continue
        null_share = feature["n_null"] / feature["n_values"]
        if null_share > feature["null_threshold"]:
            problems.append(
                f"{name}: {null_share:.1%} of values are null, "
                f"allowed {feature['null_threshold']:.1%}"
            )
        # min and max are None when every value is null
        if feature["expected_min"] is not None and feature["min_value"] is not None:
            if feature["min_value"] < feature["expected_min"]:
                problems.append(
                    f"{name}: minimum {feature['min_value']:g} is below "
                    f"the expected {feature['expected_min']:g}"
                )
        if feature["expected_max"] is not None and feature["max_value"] is not None:
            if feature["max_value"] > feature["expected_max"]:
                problems.append(
                    f"{name}: maximum {feature['max_value']:g} is above "
                    f"the expected {feature['expected_max']:g}"
                )
    return problems
