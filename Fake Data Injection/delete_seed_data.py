"""
EduSense — Fake Data Cleaner
==============================
Deletes ALL rows injected by seed_data.py using seed_ids.json.
Deletion order respects foreign key constraints.

Run:
    python delete_seed_data.py

Flags:
    --dry-run    Show what would be deleted without touching the DB
    --force      Skip the confirmation prompt
"""

import os, json, sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

sb = create_client(
    os.getenv('SUPABASE_URL', ''),
    os.getenv('SUPABASE_SERVICE_KEY', ''),
)

SEED_FILE = Path('seed_ids.json')

DRY_RUN = '--dry-run' in sys.argv
FORCE   = '--force'   in sys.argv


def delete_rows(table, ids, id_col='id'):
    """Delete rows in batches of 50 (Supabase IN clause limit)."""
    if not ids:
        return 0
    deleted = 0
    batch_size = 50
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        if not DRY_RUN:
            try:
                sb.table(table).delete().in_(id_col, batch).execute()
                deleted += len(batch)
            except Exception as e:
                print(f'  ⚠️  {table}: {e}')
        else:
            deleted += len(batch)
    return deleted


def delete_auth_users(user_ids):
    """Delete Supabase auth users via admin API."""
    deleted = 0
    for uid in user_ids:
        if not DRY_RUN:
            try:
                sb.auth.admin.delete_user(uid)
                deleted += 1
            except Exception as e:
                print(f'  ⚠️  Auth user {uid[:8]}...: {e}')
        else:
            deleted += 1
    return deleted


def main():
    print('═' * 55)
    print('  EduSense — Fake Data Cleaner')
    if DRY_RUN:
        print('  MODE: DRY RUN — no changes will be made')
    print('═' * 55)

    if not SEED_FILE.exists():
        print(f'\n❌ {SEED_FILE} not found.')
        print('   Nothing to delete — seed_data.py has not been run yet.')
        return

    ids = json.loads(SEED_FILE.read_text())

    # Show summary of what will be deleted
    print('\nWill delete:')
    print(f'  Notebooks:        {len(ids.get("notebooks", []))}')
    print(f'  Struggle moments: {len(ids.get("struggle_moments", []))}')
    print(f'  Emotion history:  {len(ids.get("emotion_history", []))}')
    print(f'  Sessions:         {len(ids.get("sessions", []))}')
    print(f'  Subjects:         {len(ids.get("subjects", []))}')
    print(f'  Profiles:         {len(ids.get("profiles", []))}')
    print(f'  Auth users:       {len(ids.get("auth_users", []))}')

    if not DRY_RUN and not FORCE:
        ans = input('\nProceed? [y/N] ').strip().lower()
        if ans != 'y':
            print('Aborted.')
            return

    print()

    # ── Delete in reverse dependency order ────────────────────────
    # (child rows first, then parents)

    n = delete_rows('notebooks',        ids.get('notebooks', []))
    print(f'  {"[DRY]" if DRY_RUN else "✅"} Notebooks:        {n}')

    n = delete_rows('struggle_moments', ids.get('struggle_moments', []))
    print(f'  {"[DRY]" if DRY_RUN else "✅"} Struggle moments: {n}')

    n = delete_rows('emotion_history',  ids.get('emotion_history', []))
    print(f'  {"[DRY]" if DRY_RUN else "✅"} Emotion history:  {n}')

    n = delete_rows('sessions',         ids.get('sessions', []))
    print(f'  {"[DRY]" if DRY_RUN else "✅"} Sessions:         {n}')

    n = delete_rows('subjects',         ids.get('subjects', []))
    print(f'  {"[DRY]" if DRY_RUN else "✅"} Subjects:         {n}')

    n = delete_rows('profiles',         ids.get('profiles', []))
    print(f'  {"[DRY]" if DRY_RUN else "✅"} Profiles:         {n}')

    n = delete_auth_users(ids.get('auth_users', []))
    print(f'  {"[DRY]" if DRY_RUN else "✅"} Auth users:       {n}')

    if not DRY_RUN:
        SEED_FILE.unlink()
        print(f'\n  🗑️  Deleted {SEED_FILE}')

    print('\n═' * 55)
    if DRY_RUN:
        print('  Dry run complete — nothing was changed.')
    else:
        print('  ✅ All fake data removed. Database is clean.')
    print('═' * 55)


if __name__ == '__main__':
    main()
