-- Source event tables. Filled by `python -m feature_store.datagen`.

CREATE TABLE IF NOT EXISTS raw.users (
    user_id    BIGINT PRIMARY KEY,
    signup_ts  TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS raw.products (
    product_id  BIGINT PRIMARY KEY,
    category    TEXT NOT NULL,
    price       NUMERIC(10, 2) NOT NULL
);

CREATE TABLE IF NOT EXISTS raw.product_views (
    view_id     BIGINT PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES raw.users (user_id),
    product_id  BIGINT NOT NULL REFERENCES raw.products (product_id),
    view_ts     TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS raw.orders (
    order_id      BIGINT PRIMARY KEY,
    user_id       BIGINT NOT NULL REFERENCES raw.users (user_id),
    order_ts      TIMESTAMPTZ NOT NULL,
    total_amount  NUMERIC(12, 2) NOT NULL
);

CREATE TABLE IF NOT EXISTS raw.order_items (
    order_id    BIGINT NOT NULL REFERENCES raw.orders (order_id),
    product_id  BIGINT NOT NULL REFERENCES raw.products (product_id),
    quantity    INT NOT NULL CHECK (quantity > 0),
    unit_price  NUMERIC(10, 2) NOT NULL,
    PRIMARY KEY (order_id, product_id)
);

-- Feature SQL filters by entity and time window
CREATE INDEX IF NOT EXISTS idx_product_views_user_ts ON raw.product_views (user_id, view_ts);
CREATE INDEX IF NOT EXISTS idx_product_views_product_ts ON raw.product_views (product_id, view_ts);
CREATE INDEX IF NOT EXISTS idx_orders_user_ts ON raw.orders (user_id, order_ts);
