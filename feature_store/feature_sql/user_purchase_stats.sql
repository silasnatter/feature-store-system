WITH order_stats AS (
    -- No lower bound here: days_since_last_order needs the full history,
    -- so the 30-day features use FILTER instead
    SELECT
        orders.user_id,
        COUNT(*) FILTER (
            WHERE orders.order_ts >= %(as_of)s - INTERVAL '30 days'
        ) AS order_count_30d,
        AVG(orders.total_amount) FILTER (
            WHERE orders.order_ts >= %(as_of)s - INTERVAL '30 days'
        ) AS avg_order_value_30d,
        EXTRACT(EPOCH FROM %(as_of)s - MAX(orders.order_ts)) / 86400 AS days_since_last_order
    FROM raw.orders AS orders
    WHERE orders.order_ts < %(as_of)s
    GROUP BY orders.user_id
),

view_stats AS (
    SELECT
        views.user_id,
        COUNT(*) AS view_count_7d
    FROM raw.product_views AS views
    WHERE views.view_ts < %(as_of)s
      AND views.view_ts >= %(as_of)s - INTERVAL '7 days'
    GROUP BY views.user_id
)

SELECT
    users.user_id AS entity_id,
    COALESCE(order_stats.order_count_30d, 0) AS order_count_30d,
    order_stats.avg_order_value_30d,
    order_stats.days_since_last_order,
    COALESCE(view_stats.view_count_7d, 0) AS view_count_7d
FROM raw.users AS users
LEFT JOIN order_stats
    ON users.user_id = order_stats.user_id
LEFT JOIN view_stats
    ON users.user_id = view_stats.user_id
WHERE users.signup_ts < %(as_of)s
