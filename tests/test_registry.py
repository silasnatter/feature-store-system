from datetime import timedelta

import pytest

from feature_store.registry import (
    FeatureDefinition,
    FeatureRegistry,
    FeatureStatus,
    FeatureView,
    NotRegisteredError,
)

USER_VIEW = FeatureView(
    view_name="user_purchase_stats",
    entity_type="user",
    description="Rolling purchase aggregates per user",
    owner="data_team",
    ttl=timedelta(days=2),
)


@pytest.fixture
def registry(conn):
    registry = FeatureRegistry(conn)
    registry.register_view(USER_VIEW)
    return registry


def test_view_round_trip(registry):
    assert registry.get_view("user_purchase_stats") == USER_VIEW


def test_register_view_twice_updates_in_place(registry):
    first_id = registry.register_view(USER_VIEW)
    changed = FeatureView(view_name="user_purchase_stats", entity_type="user", owner="ml_team")

    second_id = registry.register_view(changed)

    assert second_id == first_id
    assert registry.get_view("user_purchase_stats") == changed


def test_feature_round_trip(registry):
    feature = FeatureDefinition(
        feature_name="avg_order_value_30d",
        view_name="user_purchase_stats",
        description="Mean order total over the last 30 days",
        expected_min=0,
        expected_max=10_000,
        null_threshold=0.5,
        freshness_hours=48,
    )

    registry.register_feature(feature)

    assert registry.get_feature("avg_order_value_30d") == feature


def test_register_feature_twice_updates_in_place(registry):
    feature = FeatureDefinition(feature_name="order_count_30d", view_name="user_purchase_stats")
    first_id = registry.register_feature(feature)
    deprecated = FeatureDefinition(
        feature_name="order_count_30d",
        view_name="user_purchase_stats",
        dtype="int",
        status=FeatureStatus.DEPRECATED,
    )

    second_id = registry.register_feature(deprecated)

    assert second_id == first_id
    assert registry.get_feature("order_count_30d") == deprecated


def test_register_feature_for_unknown_view_fails(registry):
    feature = FeatureDefinition(feature_name="orphan", view_name="no_such_view")

    with pytest.raises(NotRegisteredError, match="no_such_view"):
        registry.register_feature(feature)


def test_unknown_names_raise(registry):
    with pytest.raises(NotRegisteredError):
        registry.get_feature("no_such_feature")
    with pytest.raises(NotRegisteredError):
        registry.get_view("no_such_view")


def test_list_features_filters_by_status_and_view(registry):
    registry.register_view(FeatureView(view_name="product_stats", entity_type="product"))
    registry.register_feature(FeatureDefinition("order_count_30d", "user_purchase_stats"))
    registry.register_feature(
        FeatureDefinition("old_feature", "user_purchase_stats", status=FeatureStatus.DEPRECATED)
    )
    registry.register_feature(FeatureDefinition("view_count_7d", "product_stats"))

    def names(**filters):
        return [feature.feature_name for feature in registry.list_features(**filters)]

    assert names() == ["order_count_30d", "view_count_7d"]
    assert names(status=None) == ["old_feature", "order_count_30d", "view_count_7d"]
    assert names(status=FeatureStatus.DEPRECATED) == ["old_feature"]
    assert names(view_name="product_stats") == ["view_count_7d"]
