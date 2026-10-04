"""Command line for the feature store.

python -m feature_store apply
python -m feature_store compute --view user_purchase_stats --as-of 2026-06-01
python -m feature_store backfill --view user_purchase_stats --start 2026-01-02 --end 2026-09-28
python -m feature_store validate --view user_purchase_stats --as-of 2026-09-28
python -m feature_store materialize --view user_purchase_stats --as-of 2026-09-28
python -m feature_store training-set --start 2026-02-01 --end 2026-08-01 --out data/training_set.csv
python -m feature_store train --start 2026-02-01 --end 2026-08-01 --test-from 2026-07-01
"""

import argparse
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from feature_store import definitions
from feature_store.compute import backfill, compute_features
from feature_store.config import get_settings
from feature_store.db import connect
from feature_store.online import connect_redis, materialize, materialized_until
from feature_store.registry import FeatureRegistry
from feature_store.validation import validate_snapshot


def _as_of(value: str) -> datetime:
    """Parse a date or timestamp; without an offset it is read as UTC."""
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m feature_store")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("apply", help="write the feature definitions to the registry")

    compute = commands.add_parser("compute", help="compute a feature view as of one moment")
    compute.add_argument("--view", required=True)
    compute.add_argument("--as-of", required=True, type=_as_of)

    fill = commands.add_parser("backfill", help="compute a feature view for every day in a range")
    fill.add_argument("--view", required=True)
    fill.add_argument("--start", required=True, type=date.fromisoformat)
    fill.add_argument("--end", required=True, type=date.fromisoformat)

    validate = commands.add_parser("validate", help="check the values a view has for one moment")
    validate.add_argument("--view", required=True)
    validate.add_argument("--as-of", required=True, type=_as_of)

    online = commands.add_parser("materialize", help="copy the latest values of a view to Redis")
    online.add_argument("--view", required=True)
    online.add_argument("--as-of", type=_as_of, default=None, help="default: now")
    online.add_argument(
        "--force", action="store_true", help="overwrite even if Redis holds later values"
    )

    training = commands.add_parser("training-set", help="write labels and features to a CSV")
    training.add_argument("--start", required=True, type=date.fromisoformat)
    training.add_argument("--end", required=True, type=date.fromisoformat)
    training.add_argument("--out", required=True, type=Path)

    train = commands.add_parser("train", help="train the purchase model and store it in MLflow")
    train.add_argument("--start", required=True, type=date.fromisoformat)
    train.add_argument("--end", required=True, type=date.fromisoformat)
    train.add_argument("--test-from", required=True, type=_as_of, help="first test snapshot")

    args = parser.parse_args()
    with connect() as conn:
        if args.command == "apply":
            definitions.apply(FeatureRegistry(conn))
            print(f"Applied {len(definitions.VIEWS)} views, {len(definitions.FEATURES)} features")
        elif args.command == "compute":
            written = compute_features(conn, args.view, args.as_of)
            print(f"{args.view} as of {args.as_of.isoformat()}: {written:,} values written")
        elif args.command == "backfill":
            written = backfill(conn, args.view, args.start, args.end)
            print(f"{args.view} {args.start} to {args.end}: {written:,} values written")
        elif args.command == "validate":
            problems = validate_snapshot(conn, args.view, args.as_of)
            if problems:
                print(f"{args.view} as of {args.as_of.isoformat()}: {len(problems)} problems")
                for problem in problems:
                    print(f"  {problem}")
                # A non-zero exit status tells the caller (Airflow) that the check failed
                sys.exit(1)
            print(f"{args.view} as of {args.as_of.isoformat()}: ok")
        elif args.command == "materialize":
            as_of = args.as_of or datetime.now(UTC)
            client = connect_redis()
            entities = materialize(conn, client, args.view, as_of, force=args.force)
            if entities is None:
                until = materialized_until(client, args.view)
                print(
                    f"{args.view} as of {as_of.isoformat()}: skipped, Redis already holds "
                    f"values as of {until.isoformat()} (--force overwrites them)"
                )
            else:
                print(f"{args.view} as of {as_of.isoformat()}: {entities:,} entities in Redis")
        elif args.command in ("training-set", "train"):
            # Imported here: these pull in pandas, scikit-learn and MLflow, which
            # the daily commands above should not have to load.
            from feature_store.modeling import MODEL_FEATURES, train_and_register
            from feature_store.training import build_training_set, monthly_snapshots

            snapshots = monthly_snapshots(args.start, args.end)
            if args.command == "training-set":
                feature_names = [feature.feature_name for feature in definitions.FEATURES]
                frame = build_training_set(conn, snapshots, feature_names)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(args.out, index=False)
                print(f"{len(frame):,} rows from {len(snapshots)} snapshots written to {args.out}")
            else:
                frame = build_training_set(conn, snapshots, MODEL_FEATURES)
                result = train_and_register(
                    frame, args.test_from, get_settings().mlflow_tracking_uri
                )
                print(f"Trained on {result.train_rows:,} rows, tested on {result.test_rows:,}")
                for name, value in result.metrics.items():
                    print(f"  {name:<15} {value:.3f}")
                print(f"Registered as purchase_model version {result.version}, now the champion")


if __name__ == "__main__":
    main()
