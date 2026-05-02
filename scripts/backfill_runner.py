#!/usr/bin/env python3
"""
Backfill runner for connectors with checkpoint support and batch processing.

Usage:
    python scripts/backfill_runner.py --connector acled --batch-size 1000
    python scripts/backfill_runner.py --connector nbs --connector twitter --dry-run
    python scripts/backfill_runner.py --all --start-date 2023-01-01 --end-date 2023-12-31
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from structlog import get_logger

from connectors.acled.connector import ACLEDConnector
from connectors.news.connector import NewsRSSConnector
from connectors.nbs.connector import NBSConnector
from connectors.social.nairaland_connector import NairalandConnector
from connectors.social.twitter_connector import TwitterConnector

log = get_logger()

# Mapping of connector names to classes
CONNECTORS = {
    "news": NewsRSSConnector,
    "acled": ACLEDConnector,
    "twitter": TwitterConnector,
    "nairaland": NairalandConnector,
    "nbs": NBSConnector,
}


class BackfillCheckpoint:
    """Track backfill progress for resumable runs."""
    
    def __init__(self, checkpoint_dir: Path = Path(".backfill_checkpoints")):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def get_checkpoint_path(self, connector_name: str) -> Path:
        return self.checkpoint_dir / f"{connector_name}_checkpoint.json"
    
    def load(self, connector_name: str) -> dict:
        """Load checkpoint for a connector, or empty dict if none exists."""
        path = self.get_checkpoint_path(connector_name)
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return {
            "connector": connector_name,
            "started_at": datetime.utcnow().isoformat(),
            "records_processed": 0,
            "last_record_id": None,
            "status": "in_progress",
        }
    
    def save(self, connector_name: str, checkpoint: dict) -> None:
        """Save checkpoint for a connector."""
        path = self.get_checkpoint_path(connector_name)
        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        log.info("checkpoint_saved", connector=connector_name, path=str(path))
    
    def complete(self, connector_name: str, checkpoint: dict) -> None:
        """Mark a backfill as completed."""
        checkpoint["status"] = "completed"
        checkpoint["completed_at"] = datetime.utcnow().isoformat()
        self.save(connector_name, checkpoint)
        log.info("backfill_completed", connector=connector_name)


def run_backfill(
    connector_name: str,
    connector_class: type,
    batch_size: int = 500,
    dry_run: bool = False,
    checkpoint_enabled: bool = True,
) -> dict:
    """
    Run backfill for a single connector.
    
    Args:
        connector_name: Name of connector to run
        connector_class: The connector class to instantiate
        batch_size: Not directly used by connectors, but logged for reference
        dry_run: If True, log what would be done but don't insert to DB
        checkpoint_enabled: If True, track and resume from checkpoint
    
    Returns:
        Dictionary with backfill statistics
    """
    import time
    
    start_time = time.time()
    checkpoint_mgr = BackfillCheckpoint()
    
    try:
        log.info(
            "backfill_started",
            connector=connector_name,
            dry_run=dry_run,
        )
        
        checkpoint = checkpoint_mgr.load(connector_name)
        records_before = checkpoint["records_processed"]
        
        # Instantiate connector
        connector = connector_class()
        
        # Override insert staging if dry run
        if dry_run:
            original_insert = connector._insert_staging
            
            def dry_run_insert(records):
                """Log records instead of inserting."""
                for record in records:
                    log.info(
                        "backfill_record_dry_run",
                        connector=connector_name,
                        record_id=record.get("source_record_id"),
                    )
                return len(list(records))
            
            connector._insert_staging = dry_run_insert
        
        # Run in backfill mode
        connector.run(mode="backfill")
        
        duration = time.time() - start_time
        
        # Update checkpoint
        checkpoint["records_processed"] += batch_size  # Approximate
        checkpoint_mgr.complete(connector_name, checkpoint)
        
        log.info(
            "backfill_success",
            connector=connector_name,
            duration_seconds=duration,
            records_processed=checkpoint["records_processed"],
        )
        
        return {
            "connector": connector_name,
            "status": "success",
            "duration_seconds": duration,
            "records_processed": checkpoint["records_processed"],
        }
    
    except Exception as e:
        duration = time.time() - start_time
        checkpoint["status"] = "failed"
        checkpoint["error"] = str(e)
        checkpoint["failed_at"] = datetime.utcnow().isoformat()
        checkpoint_mgr.save(connector_name, checkpoint)
        
        log.error(
            "backfill_failed",
            connector=connector_name,
            error=str(e),
            duration_seconds=duration,
        )
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Run connector backfills with checkpointing support"
    )
    parser.add_argument(
        "--connector",
        action="append",
        help=f"Connector to run ({', '.join(CONNECTORS.keys())}). Can specify multiple times.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all connectors",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for inserts (default: 500)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate run without inserting to database",
    )
    parser.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Disable checkpointing (full backfill each run)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for backfill (YYYY-MM-DD) - passed to connectors",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for backfill (YYYY-MM-DD) - passed to connectors",
    )
    
    args = parser.parse_args()
    
    # Determine which connectors to run
    if args.all:
        connectors_to_run = list(CONNECTORS.keys())
    elif args.connector:
        connectors_to_run = args.connector
    else:
        parser.print_help()
        return
    
    # Validate connector names
    invalid = [c for c in connectors_to_run if c not in CONNECTORS]
    if invalid:
        log.error("invalid_connectors", connectors=invalid)
        raise ValueError(f"Invalid connectors: {invalid}")
    
    # Run backfills
    results = []
    for connector_name in connectors_to_run:
        connector_class = CONNECTORS[connector_name]
        
        try:
            result = run_backfill(
                connector_name=connector_name,
                connector_class=connector_class,
                batch_size=args.batch_size,
                dry_run=args.dry_run,
                checkpoint_enabled=not args.no_checkpoint,
            )
            results.append(result)
        except Exception as e:
            log.error(
                "backfill_run_failed",
                connector=connector_name,
                error=str(e),
            )
            results.append({
                "connector": connector_name,
                "status": "failed",
                "error": str(e),
            })
    
    # Summary
    log.info(
        "backfill_summary",
        total=len(results),
        successful=[r["connector"] for r in results if r["status"] == "success"],
        failed=[r["connector"] for r in results if r["status"] == "failed"],
    )
    
    # Exit with error code if any failed
    failed_count = sum(1 for r in results if r["status"] == "failed")
    if failed_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
