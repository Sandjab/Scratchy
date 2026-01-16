#!/usr/bin/env python3
"""CLI for database backup management."""

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scratchy.config import get_settings


def create_backup(args):
    """Create a database backup."""
    settings = get_settings()
    db_path = Path(settings.storage.db_path)
    backup_dir = Path(settings.storage.backup_dir)

    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}")
        sys.exit(1)

    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_name = f"scratchy_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.db"
    backup_path = backup_dir / backup_name

    try:
        shutil.copy2(db_path, backup_path)
    except Exception as e:
        print(f"Error creating backup: {e}")
        sys.exit(1)

    size_mb = backup_path.stat().st_size / (1024 * 1024)
    print(f"Backup created: {backup_path}")
    print(f"Size: {size_mb:.2f} MB")


def list_backups(args):
    """List available backups."""
    settings = get_settings()
    backup_dir = Path(settings.storage.backup_dir)

    if not backup_dir.exists():
        print("No backups found.")
        return

    backups = sorted(backup_dir.glob("scratchy_*.db"), reverse=True)

    if not backups:
        print("No backups found.")
        return

    print(f"\n{'Backup File':<40} {'Size':<12} {'Created'}")
    print("-" * 70)

    for backup in backups:
        size_mb = backup.stat().st_size / (1024 * 1024)
        created = datetime.fromtimestamp(backup.stat().st_mtime)
        print(f"{backup.name:<40} {size_mb:>8.2f} MB  {created.strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\nTotal: {len(backups)} backups\n")


def restore_backup(args):
    """Restore from a backup."""
    settings = get_settings()
    db_path = Path(settings.storage.db_path)
    backup_dir = Path(settings.storage.backup_dir)
    backup_path = backup_dir / args.backup_name

    if not backup_path.exists():
        print(f"Error: Backup not found: {backup_path}")
        sys.exit(1)

    if db_path.exists() and not args.yes:
        confirm = input("This will overwrite the current database. Continue? [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return

    try:
        # Create a backup of current database first
        if db_path.exists():
            pre_restore = db_path.with_suffix(".db.pre_restore")
            shutil.copy2(db_path, pre_restore)
            print(f"Current database backed up to: {pre_restore}")

        shutil.copy2(backup_path, db_path)
        print(f"Database restored from: {backup_path}")
    except Exception as e:
        print(f"Error restoring backup: {e}")
        sys.exit(1)


def cleanup_backups(args):
    """Clean up old backups based on retention policy."""
    settings = get_settings()
    backup_dir = Path(settings.storage.backup_dir)
    retention_days = args.days or settings.storage.backup_retention_days

    if not backup_dir.exists():
        print("No backups to clean up.")
        return

    cutoff = datetime.utcnow().timestamp() - (retention_days * 24 * 60 * 60)
    backups = list(backup_dir.glob("scratchy_*.db"))
    deleted = 0

    for backup in backups:
        if backup.stat().st_mtime < cutoff:
            if args.dry_run:
                print(f"Would delete: {backup.name}")
            else:
                backup.unlink()
                print(f"Deleted: {backup.name}")
            deleted += 1

    if args.dry_run:
        print(f"\nWould delete {deleted} backups (dry run)")
    else:
        print(f"\nDeleted {deleted} backups")


def main():
    parser = argparse.ArgumentParser(
        description="Scratchy Database Backup Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create backup
    create_parser = subparsers.add_parser("create", help="Create a new backup")
    create_parser.set_defaults(func=create_backup)

    # List backups
    list_parser = subparsers.add_parser("list", help="List available backups")
    list_parser.set_defaults(func=list_backups)

    # Restore backup
    restore_parser = subparsers.add_parser("restore", help="Restore from a backup")
    restore_parser.add_argument("backup_name", help="Backup filename to restore")
    restore_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    restore_parser.set_defaults(func=restore_backup)

    # Cleanup backups
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old backups")
    cleanup_parser.add_argument("--days", "-d", type=int, help="Retention days (overrides config)")
    cleanup_parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    cleanup_parser.set_defaults(func=cleanup_backups)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
