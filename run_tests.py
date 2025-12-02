#!/usr/bin/env python3
"""
Test runner script for RAG Service
Runs all unit tests and generates a coverage report
"""

import sys
import subprocess
from pathlib import Path


def run_tests(verbose=True, coverage=False):
    """
    Run pytest tests.

    Args:
        verbose: Show verbose output
        coverage: Generate coverage report
    """
    # Get project root
    project_root = Path(__file__).parent

    # Build pytest command
    cmd = ["pytest", "tests/test_part_a.py"]

    if verbose:
        cmd.append("-v")

    if coverage:
        cmd.extend([
            "--cov=part_a",
            "--cov-report=term-missing",
            "--cov-report=html"
        ])

    # Add color output
    cmd.append("--color=yes")

    # Run tests
    print("=" * 70)
    print("Running RAG Service Tests")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70)
    print()

    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            check=False
        )

        print()
        print("=" * 70)
        if result.returncode == 0:
            print("[SUCCESS] ALL TESTS PASSED")
        else:
            print(f"[FAILED] TESTS FAILED (exit code: {result.returncode})")
        print("=" * 70)

        return result.returncode

    except FileNotFoundError:
        print("ERROR: pytest not found. Install with: pip install pytest")
        return 1
    except Exception as e:
        print(f"ERROR running tests: {e}")
        return 1


def main():
    """Main entry point."""
    # Parse simple arguments
    verbose = "-v" in sys.argv or "--verbose" in sys.argv
    coverage = "-c" in sys.argv or "--coverage" in sys.argv
    help_flag = "-h" in sys.argv or "--help" in sys.argv

    if help_flag:
        print("Usage: python run_tests.py [options]")
        print()
        print("Options:")
        print("  -v, --verbose    Show verbose test output")
        print("  -c, --coverage   Generate coverage report")
        print("  -h, --help       Show this help message")
        print()
        print("Examples:")
        print("  python run_tests.py              # Run tests")
        print("  python run_tests.py -v           # Run with verbose output")
        print("  python run_tests.py -v -c        # Run with coverage")
        return 0

    return run_tests(verbose=verbose or True, coverage=coverage)


if __name__ == "__main__":
    sys.exit(main())
