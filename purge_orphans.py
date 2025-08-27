import argparse
from src.io_utils import delete_orphaned_runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without actually deleting")
    parser.add_argument("--direction", choices=["db_only", "files_only", "both"], default="both", 
                       help="Direction to purge: db_only (remove DB entries for missing files), files_only (remove untracked files), both (default)")
    args = parser.parse_args()
    delete_orphaned_runs(dry_run=args.dry_run, direction=args.direction)