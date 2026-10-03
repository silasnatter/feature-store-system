from dataclasses import asdict, dataclass
from datetime import timedelta
from enum import StrEnum

import psycopg


class FeatureStatus(StrEnum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    TESTING = "testing"


class NotRegisteredError(LookupError):
    """Raised when a feature view or feature is not in the registry."""


@dataclass(frozen=True)
class FeatureView:
    """A group of features that share an entity and are computed together."""

    view_name: str
    entity_type: str
    description: str | None = None
    owner: str | None = None
    ttl: timedelta | None = None
    source_sql: str | None = None


@dataclass(frozen=True)
class FeatureDefinition:
    feature_name: str
    view_name: str
    description: str | None = None
    dtype: str = "float"
    status: FeatureStatus = FeatureStatus.ACTIVE

    # Monitoring thresholds
    expected_min: float | None = None
    expected_max: float | None = None
    null_threshold: float = 0.01
    freshness_hours: int = 24


_FEATURE_COLUMNS = """
    fd.feature_name, fv.view_name, fd.description, fd.dtype, fd.status,
    fd.expected_min, fd.expected_max, fd.null_threshold, fd.freshness_hours
"""


class FeatureRegistry:
    """Metadata for feature views and features.

    Expects a connection with `dict_row` rows (see `feature_store.db`). Does not
    commit; the caller owns the transaction.
    """

    def __init__(self, conn: psycopg.Connection):
        self.conn = conn

    def register_view(self, view: FeatureView) -> int:
        """Create the view, or update it if the name already exists. Returns its id."""
        row = self.conn.execute(
            """
            INSERT INTO feature_store.feature_views
                (view_name, entity_type, description, owner, ttl, source_sql)
            VALUES
                (%(view_name)s, %(entity_type)s, %(description)s, %(owner)s,
                 %(ttl)s, %(source_sql)s)
            ON CONFLICT (view_name) DO UPDATE SET
                entity_type = EXCLUDED.entity_type,
                description = EXCLUDED.description,
                owner = EXCLUDED.owner,
                ttl = EXCLUDED.ttl,
                source_sql = EXCLUDED.source_sql
            RETURNING view_id
            """,
            asdict(view),
        ).fetchone()
        return row["view_id"]

    def get_view(self, view_name: str) -> FeatureView:
        row = self.conn.execute(
            """
            SELECT view_name, entity_type, description, owner, ttl, source_sql
            FROM feature_store.feature_views
            WHERE view_name = %s
            """,
            (view_name,),
        ).fetchone()
        if row is None:
            raise NotRegisteredError(f"Feature view '{view_name}' is not registered")
        return FeatureView(**row)

    def register_feature(self, feature: FeatureDefinition) -> int:
        """Create the feature, or update it if the name already exists. Returns its id."""
        row = self.conn.execute(
            """
            INSERT INTO feature_store.feature_definitions
                (view_id, feature_name, description, dtype, status,
                 expected_min, expected_max, null_threshold, freshness_hours)
            SELECT
                view_id, %(feature_name)s, %(description)s, %(dtype)s, %(status)s,
                %(expected_min)s, %(expected_max)s, %(null_threshold)s, %(freshness_hours)s
            FROM feature_store.feature_views
            WHERE view_name = %(view_name)s
            ON CONFLICT (feature_name) DO UPDATE SET
                view_id = EXCLUDED.view_id,
                description = EXCLUDED.description,
                dtype = EXCLUDED.dtype,
                status = EXCLUDED.status,
                expected_min = EXCLUDED.expected_min,
                expected_max = EXCLUDED.expected_max,
                null_threshold = EXCLUDED.null_threshold,
                freshness_hours = EXCLUDED.freshness_hours
            RETURNING feature_id
            """,
            asdict(feature),
        ).fetchone()
        if row is None:
            raise NotRegisteredError(f"Feature view '{feature.view_name}' is not registered")
        return row["feature_id"]

    def get_feature(self, feature_name: str) -> FeatureDefinition:
        row = self.conn.execute(
            f"""
            SELECT {_FEATURE_COLUMNS}
            FROM feature_store.feature_definitions fd
            JOIN feature_store.feature_views fv USING (view_id)
            WHERE fd.feature_name = %s
            """,
            (feature_name,),
        ).fetchone()
        if row is None:
            raise NotRegisteredError(f"Feature '{feature_name}' is not registered")
        return _to_feature(row)

    def list_features(
        self,
        status: FeatureStatus | None = FeatureStatus.ACTIVE,
        view_name: str | None = None,
    ) -> list[FeatureDefinition]:
        """List features, by default only active ones. Pass status=None for all."""
        rows = self.conn.execute(
            f"""
            SELECT {_FEATURE_COLUMNS}
            FROM feature_store.feature_definitions fd
            JOIN feature_store.feature_views fv USING (view_id)
            WHERE (%(status)s::text IS NULL OR fd.status = %(status)s)
              AND (%(view_name)s::text IS NULL OR fv.view_name = %(view_name)s)
            ORDER BY fd.feature_name
            """,
            {"status": status, "view_name": view_name},
        ).fetchall()
        return [_to_feature(row) for row in rows]


def _to_feature(row: dict) -> FeatureDefinition:
    return FeatureDefinition(**{**row, "status": FeatureStatus(row["status"])})
