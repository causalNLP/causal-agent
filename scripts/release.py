#!/usr/bin/env python3
"""
Release management script for causal-agent.

This script helps manage version updates and releases.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Tuple


def get_current_version() -> str:
    """Get the current version from causal_agent/__init__.py"""
    init_file = Path("causal_agent/__init__.py")
    if not init_file.exists():
        raise FileNotFoundError("causal_agent/__init__.py not found")
    
    content = init_file.read_text()
    match = re.search(r'__version__\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError("Version not found in causal_agent/__init__.py")
    
    return match.group(1)


def update_version(new_version: str) -> None:
    """Update version in all relevant files"""
    print(f"Updating version to {new_version}...")
    
    # Update causal_agent/__init__.py
    init_file = Path("causal_agent/__init__.py")
    content = init_file.read_text()
    updated_content = re.sub(
        r'__version__\s*=\s*"[^"]+"',
        f'__version__ = "{new_version}"',
        content
    )
    init_file.write_text(updated_content)
    print(f"✅ Updated {init_file}")
    
    # Update pyproject.toml
    pyproject_file = Path("pyproject.toml")
    if pyproject_file.exists():
        content = pyproject_file.read_text()
        updated_content = re.sub(
            r'version\s*=\s*"[^"]+"',
            f'version = "{new_version}"',
            content
        )
        pyproject_file.write_text(updated_content)
        print(f"✅ Updated {pyproject_file}")


def validate_version_format(version: str) -> bool:
    """Validate version follows semantic versioning"""
    pattern = r'^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$'
    return bool(re.match(pattern, version))


def increment_version(current_version: str, increment_type: str) -> str:
    """Increment version based on type (major, minor, patch)"""
    parts = current_version.split('.')
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {current_version}")
    
    major, minor, patch = map(int, parts)
    
    if increment_type == "major":
        return f"{major + 1}.0.0"
    elif increment_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif increment_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Invalid increment type: {increment_type}")


def run_command(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def create_git_tag(version: str, push: bool = False) -> None:
    """Create and optionally push a git tag"""
    tag = f"v{version}"
    
    # Check if tag already exists
    result = run_command(["git", "tag", "-l", tag], check=False)
    if tag in result.stdout:
        print(f"⚠️  Tag {tag} already exists")
        return
    
    # Create tag
    run_command(["git", "tag", "-a", tag, "-m", f"Release version {version}"])
    print(f"✅ Created tag {tag}")
    
    if push:
        run_command(["git", "push", "origin", tag])
        print(f"✅ Pushed tag {tag}")


def check_git_status() -> bool:
    """Check if git working directory is clean"""
    result = run_command(["git", "status", "--porcelain"], check=False)
    return len(result.stdout.strip()) == 0


def main():
    parser = argparse.ArgumentParser(description="Release management for causal-agent")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Current version command
    current_parser = subparsers.add_parser("current", help="Show current version")
    
    # Update version command
    update_parser = subparsers.add_parser("update", help="Update version")
    update_group = update_parser.add_mutually_exclusive_group(required=True)
    update_group.add_argument("--version", help="Specific version to set")
    update_group.add_argument("--increment", choices=["major", "minor", "patch"], 
                             help="Increment version type")
    
    # Release command
    release_parser = subparsers.add_parser("release", help="Create a release")
    release_parser.add_argument("--version", help="Version to release (optional)")
    release_parser.add_argument("--increment", choices=["major", "minor", "patch"],
                               help="Increment version type")
    release_parser.add_argument("--push", action="store_true", 
                               help="Push tag to remote")
    release_parser.add_argument("--dry-run", action="store_true",
                               help="Show what would be done without doing it")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        current_version = get_current_version()
        
        if args.command == "current":
            print(f"Current version: {current_version}")
            
        elif args.command == "update":
            if args.version:
                new_version = args.version
                if not validate_version_format(new_version):
                    print(f"❌ Invalid version format: {new_version}")
                    print("Expected format: X.Y.Z or X.Y.Z-suffix")
                    sys.exit(1)
            else:
                new_version = increment_version(current_version, args.increment)
            
            print(f"Current version: {current_version}")
            print(f"New version: {new_version}")
            
            update_version(new_version)
            print("✅ Version updated successfully")
            
        elif args.command == "release":
            if not check_git_status():
                print("❌ Git working directory is not clean. Please commit or stash changes.")
                sys.exit(1)
            
            if args.version:
                new_version = args.version
                if not validate_version_format(new_version):
                    print(f"❌ Invalid version format: {new_version}")
                    sys.exit(1)
                if new_version != current_version:
                    print(f"Updating version from {current_version} to {new_version}")
                    if not args.dry_run:
                        update_version(new_version)
            elif args.increment:
                new_version = increment_version(current_version, args.increment)
                print(f"Incrementing version from {current_version} to {new_version}")
                if not args.dry_run:
                    update_version(new_version)
            else:
                new_version = current_version
                print(f"Using current version: {new_version}")
            
            print(f"Creating release for version {new_version}")
            
            if args.dry_run:
                print("🔍 Dry run - would create tag: v{new_version}")
                if args.push:
                    print("🔍 Dry run - would push tag to remote")
            else:
                # Commit version changes if any
                if args.version or args.increment:
                    run_command(["git", "add", "causal_agent/__init__.py", "pyproject.toml"])
                    run_command(["git", "commit", "-m", f"Bump version to {new_version}"])
                    print("✅ Committed version changes")
                
                create_git_tag(new_version, args.push)
                
                print(f"🎉 Release {new_version} created successfully!")
                print(f"To trigger the release workflow, push the tag:")
                print(f"git push origin v{new_version}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()