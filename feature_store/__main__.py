"""Command line for the feature store.

python -m feature_store apply
python -m feature_store compute --view user_purchase_stats --as-of 2026-06-01
python -m feature_store backfill --view user_purchase_stats --start 2026-01-02 --end 2026-09-28
"""

import argparse
from datetime import UTC, date, datetime

from feature_store import definitions
from feature_store.compute import backfill, compute_features
from feature_store.db import connect
from feature_store.registry import FeatureRegistry


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


if __name__ == "__main__":
    main()
