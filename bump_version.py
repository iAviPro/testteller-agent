#!/usr/bin/env python3
"""
Version bump utility for TestTeller RAG Agent.
This script updates the version in the single source of truth (_version.py).
"""

import argparse
import re
import sys
from pathlib import Path


def get_current_version():
    """Get the current version from _version.py"""
    version_file = Path("testteller/_version.py")
    if not version_file.exists():
        print("Error: testteller/_version.py not found")
        sys.exit(1)

    content = version_file.read_text()
    match = re.search(r'__version__ = ["\']([^"\']+)["\']', content)
    if not match:
        print("Error: Could not find version in _version.py")
        sys.exit(1)

    return match.group(1)


def set_version(new_version):
    """Set a new version in _version.py"""
    version_file = Path("testteller/_version.py")
    content = version_file.read_text()

    new_content = re.sub(
        r'__version__ = ["\']([^"\']+)["\']',
        f'__version__ = "{new_version}"',
        content
    )

    version_file.write_text(new_content)
    print(f"Version updated to {new_version}")


def bump_version(part):
    """Bump version (major, minor, or patch)"""
    current = get_current_version()
    major, minor, patch = map(int, current.split('.'))

    if part == 'major':
        major += 1
        minor = 0
        patch = 0
    elif part == 'minor':
        minor += 1
        patch = 0
    elif part == 'patch':
        patch += 1
    else:
        print(f"Error: Invalid part '{part}'. Use major, minor, or patch.")
        sys.exit(1)

    new_version = f"{major}.{minor}.{patch}"
    set_version(new_version)
    return new_version


def main():
    parser = argparse.ArgumentParser(
        description="Bump version for TestTeller RAG Agent")
    parser.add_argument(
        'action',
        choices=['show', 'set', 'bump'],
        help="Action to perform"
    )
    parser.add_argument(
        'value',
        nargs='?',
        help="Version number (for set) or part to bump (major/minor/patch for bump)"
    )

    args = parser.parse_args()

    if args.action == 'show':
        current = get_current_version()
        print(f"Current version: {current}")

    elif args.action == 'set':
        if not args.value:
            print("Error: Version number required for set action")
            sys.exit(1)

        # Validate version format
        if not re.match(r'^\d+\.\d+\.\d+$', args.value):
            print("Error: Version must be in format X.Y.Z")
            sys.exit(1)

        set_version(args.value)

    elif args.action == 'bump':
        if not args.value:
            print("Error: Part to bump required (major/minor/patch)")
            sys.exit(1)

        new_version = bump_version(args.value)
        print(f"Version bumped to {new_version}")


if __name__ == "__main__":
    main()
