-- Label: did the user place at least one order in the 30 days from as_of on?
-- The feature views read events before as_of, this reads events from as_of on,
-- so no order can be both an input and the answer.
SELECT
    users.user_id AS entity_id,
    (
        EXISTS (
            SELECT 1
            FROM raw.orders AS orders
            WHERE orders.user_id = users.user_id
              AND orders.order_ts >= %(as_of)s
              AND orders.order_ts < %(as_of)s + INTERVAL '30 days'
        )
    )::int AS label
FROM raw.users AS users
WHERE users.signup_ts < %(as_of)s
