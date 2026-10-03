"""The feature views and features this project defines.

`python -m feature_store apply` writes them to the registry.
"""

from datetime import timedelta
from pathlib import Path

from feature_store.registry import FeatureDefinition, FeatureRegistry, FeatureView

SQL_DIR = Path(__file__).parent / "feature_sql"


def _sql(view_name: str) -> str:
    return (SQL_DIR / f"{view_name}.sql").read_text()


VIEWS = [
    FeatureView(
        view_name="user_purchase_stats",
        entity_type="user",
        description="Recent ordering and browsing behaviour per user",
        owner="data_team",
        # Computed daily; two days tolerates one missed run
        ttl=timedelta(days=2),
        source_sql=_sql("user_purchase_stats"),
    ),
]

FEATURES = [
    FeatureDefinition(
        feature_name="order_count_30d",
        view_name="user_purchase_stats",
        description="Orders in the 30 days before the timestamp",
        dtype="int",
        expected_min=0,
    ),
    FeatureDefinition(
        feature_name="avg_order_value_30d",
        view_name="user_purchase_stats",
        description="Mean order total over the same 30 days; null without orders",
        expected_min=0,
        null_threshold=1.0,
    ),
    FeatureDefinition(
        feature_name="days_since_last_order",
        view_name="user_purchase_stats",
        description="Days since the last order; null if the user never ordered",
        expected_min=0,
        null_threshold=1.0,
    ),
    FeatureDefinition(
        feature_name="view_count_7d",
        view_name="user_purchase_stats",
        description="Product views in the 7 days before the timestamp",
        dtype="int",
        expected_min=0,
    ),
]


def apply(registry: FeatureRegistry) -> None:
    """Create or update every view and feature in the registry."""
    for view in VIEWS:
        registry.register_view(view)
    for feature in FEATURES:
        registry.register_feature(feature)
