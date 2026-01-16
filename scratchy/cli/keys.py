#!/usr/bin/env python3
"""CLI for API key management."""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scratchy.config import get_settings
from scratchy.models.database import init_database
from scratchy.services.auth import AuthService
from scratchy.services.credits import CreditService


def create_key(args):
    """Create a new API key."""
    settings = get_settings()
    _, Session = init_database(str(settings.storage.db_path))
    auth_service = AuthService(Session, settings.auth.default_rate_limit)

    plaintext_key, api_key = auth_service.create_key(
        name=args.name,
        credits=args.credits,
        rate_limit=args.rate_limit or settings.auth.default_rate_limit,
    )

    print("\n" + "=" * 60)
    print("NEW API KEY CREATED")
    print("=" * 60)
    print(f"\nKey ID:      {api_key.id}")
    print(f"Name:        {api_key.name}")
    print(f"Credits:     {api_key.credits}")
    print(f"Rate Limit:  {api_key.rate_limit} req/min")
    print(f"\n>>> API KEY: {plaintext_key} <<<")
    print("\n!!! SAVE THIS KEY - IT WILL NOT BE SHOWN AGAIN !!!")
    print("=" * 60 + "\n")


def list_keys(args):
    """List all API keys."""
    settings = get_settings()
    _, Session = init_database(str(settings.storage.db_path))
    auth_service = AuthService(Session, settings.auth.default_rate_limit)

    keys = auth_service.list_keys(include_inactive=args.all)

    if not keys:
        print("No API keys found.")
        return

    print(f"\n{'ID':<36} {'Name':<20} {'Credits':<8} {'Rate':<6} {'Active':<6} {'Last Used'}")
    print("-" * 100)

    for key in keys:
        last_used = key.last_used_at.strftime("%Y-%m-%d %H:%M") if key.last_used_at else "Never"
        active = "Yes" if key.is_active else "No"
        print(f"{key.id:<36} {key.name:<20} {key.credits:<8} {key.rate_limit:<6} {active:<6} {last_used}")

    print(f"\nTotal: {len(keys)} keys\n")


def show_key(args):
    """Show details of a specific API key."""
    settings = get_settings()
    _, Session = init_database(str(settings.storage.db_path))
    auth_service = AuthService(Session, settings.auth.default_rate_limit)
    credit_service = CreditService(Session)

    api_key = auth_service.get_key_by_id(args.key_id)

    if not api_key:
        print(f"Error: No API key found with ID: {args.key_id}")
        sys.exit(1)

    print(f"\nAPI Key Details")
    print("=" * 40)
    print(f"ID:          {api_key.id}")
    print(f"Name:        {api_key.name}")
    print(f"Credits:     {api_key.credits}")
    print(f"Rate Limit:  {api_key.rate_limit} req/min")
    print(f"Active:      {'Yes' if api_key.is_active else 'No'}")
    print(f"Created:     {api_key.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    if api_key.last_used_at:
        print(f"Last Used:   {api_key.last_used_at.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        print(f"Last Used:   Never")

    # Show recent transactions
    transactions = credit_service.get_transaction_history(api_key.id, limit=5)
    if transactions:
        print(f"\nRecent Transactions:")
        print("-" * 40)
        for t in transactions:
            sign = "+" if t.amount > 0 else ""
            print(f"  {t.timestamp.strftime('%Y-%m-%d %H:%M')} | {sign}{t.amount} | {t.reason}")

    print()


def update_key(args):
    """Update an API key."""
    settings = get_settings()
    _, Session = init_database(str(settings.storage.db_path))
    auth_service = AuthService(Session, settings.auth.default_rate_limit)

    api_key = auth_service.update_key(
        key_id=args.key_id,
        name=args.name,
        credits=args.credits,
        rate_limit=args.rate_limit,
        is_active=args.activate if args.activate else (False if args.deactivate else None),
    )

    if not api_key:
        print(f"Error: No API key found with ID: {args.key_id}")
        sys.exit(1)

    print(f"API key {api_key.id} updated successfully.")
    print(f"  Name:        {api_key.name}")
    print(f"  Credits:     {api_key.credits}")
    print(f"  Rate Limit:  {api_key.rate_limit}")
    print(f"  Active:      {'Yes' if api_key.is_active else 'No'}")


def add_credits(args):
    """Add credits to an API key."""
    settings = get_settings()
    _, Session = init_database(str(settings.storage.db_path))
    credit_service = CreditService(Session)

    success, new_balance = credit_service.add_credits(
        key_id=args.key_id,
        amount=args.amount,
        reason="admin_adjustment",
        description=args.reason or f"Added {args.amount} credits via CLI",
    )

    if not success:
        print(f"Error: Failed to add credits. Key may not exist.")
        sys.exit(1)

    print(f"Added {args.amount} credits to key {args.key_id}")
    print(f"New balance: {new_balance}")


def delete_key(args):
    """Delete (deactivate) an API key."""
    settings = get_settings()
    _, Session = init_database(str(settings.storage.db_path))
    auth_service = AuthService(Session, settings.auth.default_rate_limit)

    if not args.yes:
        confirm = input(f"Are you sure you want to delete key {args.key_id}? [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return

    success = auth_service.delete_key(args.key_id)

    if not success:
        print(f"Error: No API key found with ID: {args.key_id}")
        sys.exit(1)

    print(f"API key {args.key_id} has been deactivated.")


def main():
    parser = argparse.ArgumentParser(
        description="Scratchy API Key Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create key
    create_parser = subparsers.add_parser("create", help="Create a new API key")
    create_parser.add_argument("--name", "-n", required=True, help="Name for the key")
    create_parser.add_argument("--credits", "-c", type=int, default=0, help="Initial credits")
    create_parser.add_argument("--rate-limit", "-r", type=int, help="Rate limit (req/min)")
    create_parser.set_defaults(func=create_key)

    # List keys
    list_parser = subparsers.add_parser("list", help="List all API keys")
    list_parser.add_argument("--all", "-a", action="store_true", help="Include inactive keys")
    list_parser.set_defaults(func=list_keys)

    # Show key
    show_parser = subparsers.add_parser("show", help="Show details of an API key")
    show_parser.add_argument("key_id", help="API key ID")
    show_parser.set_defaults(func=show_key)

    # Update key
    update_parser = subparsers.add_parser("update", help="Update an API key")
    update_parser.add_argument("key_id", help="API key ID")
    update_parser.add_argument("--name", "-n", help="New name")
    update_parser.add_argument("--credits", "-c", type=int, help="New credit balance")
    update_parser.add_argument("--rate-limit", "-r", type=int, help="New rate limit")
    update_parser.add_argument("--activate", action="store_true", help="Activate the key")
    update_parser.add_argument("--deactivate", action="store_true", help="Deactivate the key")
    update_parser.set_defaults(func=update_key)

    # Add credits
    credits_parser = subparsers.add_parser("add-credits", help="Add credits to a key")
    credits_parser.add_argument("key_id", help="API key ID")
    credits_parser.add_argument("amount", type=int, help="Credits to add")
    credits_parser.add_argument("--reason", help="Reason for adjustment")
    credits_parser.set_defaults(func=add_credits)

    # Delete key
    delete_parser = subparsers.add_parser("delete", help="Delete (deactivate) an API key")
    delete_parser.add_argument("key_id", help="API key ID")
    delete_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")
    delete_parser.set_defaults(func=delete_key)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
