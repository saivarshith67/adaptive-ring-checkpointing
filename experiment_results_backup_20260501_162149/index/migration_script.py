#!/usr/bin/env python3
"""
Migration script for reorganizing experiment results.

This script migrates experiment directories from timestamp-based naming
(e.g., faceforensics_convergence_20260419_104954) to organized naming
(e.g., convergence_vanilla_20260419_104954).

Usage:
    python migration_script.py --dry-run    # See what would happen
    python migration_script.py --execute    # Perform actual migration
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path

# Base directory for experiments
EXPERIMENT_DIR = Path(__file__).parent.parent
INDEX_DIR = Path(__file__).parent
REGISTER_FILE = INDEX_DIR / "experiment_register.json"
SUMMARIES_FILE = EXPERIMENT_DIR / "experiment_summaries.jsonl"


def load_register():
    """Load the experiment register mapping."""
    if not REGISTER_FILE.exists():
        print(f"Error: Register file not found: {REGISTER_FILE}")
        sys.exit(1)
    
    with open(REGISTER_FILE, 'r') as f:
        return json.load(f)


def load_summaries():
    """Load experiment summaries."""
    if not SUMMARIES_FILE.exists():
        print(f"Error: Summaries file not found: {SUMMARIES_FILE}")
        sys.exit(1)
    
    summaries = []
    with open(SUMMARIES_FILE, 'r') as f:
        for line in f:
            if line.strip():
                summaries.append(json.loads(line))
    return summaries


def backup_summaries():
    """Create a backup of experiment_summaries.jsonl."""
    backup_file = SUMMARIES_FILE.with_suffix('.jsonl.backup')
    shutil.copy2(SUMMARIES_FILE, backup_file)
    print(f"Created backup: {backup_file}")
    return backup_file


def rename_directory(old_name, new_name, dry_run=True):
    """
    Rename an experiment directory.
    
    Args:
        old_name: Current directory name
        new_name: New directory name
        dry_run: If True, only print what would be done
    
    Returns:
        bool: True if successful or dry-run, False on error
    """
    old_path = EXPERIMENT_DIR / old_name
    new_path = EXPERIMENT_DIR / new_name
    
    # Validate old directory exists
    if not old_path.exists():
        if dry_run:
            print(f"Would rename (directory missing): {old_name} -> {new_name}")
            print(f"  Note: Directory not found, would skip during actual migration")
            return True  # Still return True for dry-run
        else:
            print(f"Skipping: Directory not found: {old_path}")
            return True  # Skip missing directories gracefully
    
    # Check if new directory already exists
    if new_path.exists():
        print(f"Skipping: New directory already exists: {new_path}")
        return True
    
    if dry_run:
        print(f"Would rename: {old_name} -> {new_name}")
        return True
    
    # Perform the rename
    try:
        # Try os.rename first (fast, same filesystem)
        try:
            os.rename(str(old_path), str(new_path))
            print(f"Renamed: {old_name} -> {new_name}")
            return True
        except OSError:
            # Fallback to copy + delete for cross-device moves
            print(f"Cross-device move detected, using copy method for {old_name}")
            shutil.copytree(str(old_path), str(new_path))
            
            # Verify copy was successful
            if new_path.exists() and len(list(old_path.iterdir())) == len(list(new_path.iterdir())):
                shutil.rmtree(str(old_path))
                print(f"Renamed (via copy): {old_name} -> {new_name}")
                return True
            else:
                print(f"Error: Copy verification failed for {old_name}")
                if new_path.exists():
                    shutil.rmtree(str(new_path))
                return False
    except Exception as e:
        print(f"Error renaming {old_name}: {e}")
        return False


def update_summaries(register, dry_run=True):
    """
    Update experiment_summaries.jsonl with new experiment names.
    
    Args:
        register: List of dicts with old_name and new_name
        dry_run: If True, only print what would be done
    
    Returns:
        bool: True if successful or dry-run, False on error
    """
    # Create mapping dictionary
    name_mapping = {entry['old_name']: entry['new_name'] for entry in register}
    
    if dry_run:
        print(f"Would update {SUMMARIES_FILE} with new experiment names")
        for old, new in name_mapping.items():
            print(f"  {old} -> {new}")
        return True
    
    # Backup first
    backup_summaries()
    
    # Read and update summaries
    updated_summaries = []
    try:
        with open(SUMMARIES_FILE, 'r') as f:
            for line in f:
                if line.strip():
                    summary = json.loads(line)
                    old_name = summary.get('experiment_name', '')
                    if old_name in name_mapping:
                        summary['experiment_name'] = name_mapping[old_name]
                        print(f"Updated: {old_name} -> {name_mapping[old_name]}")
                    updated_summaries.append(summary)
        
        # Write to temp file first, then replace
        temp_file = SUMMARIES_FILE.with_suffix('.jsonl.tmp')
        with open(temp_file, 'w') as f:
            for summary in updated_summaries:
                f.write(json.dumps(summary) + '\n')
        
        # Replace original with updated file
        os.replace(str(temp_file), str(SUMMARIES_FILE))
        print(f"Updated: {SUMMARIES_FILE}")
        return True
    except Exception as e:
        print(f"Error updating summaries: {e}")
        if temp_file.exists():
            temp_file.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(description='Migrate experiment directories to new naming convention')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--dry-run', action='store_true', help='Show what would be done without making changes (default)')
    group.add_argument('--execute', action='store_true', help='Execute the migration')
    
    args = parser.parse_args()
    dry_run = not args.execute
    
    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE - No changes will be made")
        print("=" * 60)
        print()
    else:
        print("=" * 60)
        print("EXECUTE MODE - Changes will be made!")
        print("=" * 60)
        print()
        confirm = input("Are you sure you want to proceed? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Migration cancelled.")
            sys.exit(0)
    
    # Load register
    print("Loading experiment register...")
    register = load_register()
    print(f"Found {len(register)} experiments to migrate")
    print()
    
    # Process each experiment
    print("Processing directories...")
    success_count = 0
    error_count = 0
    
    for entry in register:
        old_name = entry['old_name']
        new_name = entry['new_name']
        
        if rename_directory(old_name, new_name, dry_run):
            success_count += 1
        else:
            error_count += 1
    
    print()
    print(f"Directory migration: {success_count} succeeded, {error_count} errors")
    print()
    
    # Update experiment_summaries.jsonl
    print("Updating experiment_summaries.jsonl...")
    if update_summaries(register, dry_run):
        print("Summaries update: Success")
    else:
        print("Summaries update: Failed")
        error_count += 1
    
    print()
    print("=" * 60)
    if dry_run:
        print("DRY RUN COMPLETE - No changes made")
        print("Run with --execute to perform actual migration")
    else:
        print("MIGRATION COMPLETE")
        print(f"Successfully migrated {success_count} experiments")
        if error_count > 0:
            print(f"Encountered {error_count} errors")
    print("=" * 60)


if __name__ == '__main__':
    main()
