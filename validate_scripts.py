#!/usr/bin/env python3
"""
Validate that all benchmark scripts are syntactically correct.
"""

import ast
import sys


def validate_script(filename: str) -> bool:
    """Validate a Python script."""
    print(f"Validating {filename}...")
    try:
        with open(filename, 'r') as f:
            code = f.read()
        ast.parse(code)
        print(f"  ✅ {filename} is valid")
        return True
    except SyntaxError as e:
        print(f"  ❌ {filename} has syntax error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ {filename} validation failed: {e}")
        return False


def main():
    print("=" * 80)
    print("SCRIPT VALIDATION")
    print("=" * 80)
    print()
    
    scripts = [
        'download_dataset.py',
        'comprehensive_analysis.py',
        'multi_config_benchmark.py',
        'benchmark_real_sharegpt.py',
    ]
    
    all_valid = True
    for script in scripts:
        if not validate_script(script):
            all_valid = False
        print()
    
    print("=" * 80)
    if all_valid:
        print("✅ ALL SCRIPTS VALID")
        print()
        print("Usage:")
        print("  1. Download datasets:")
        print("     python download_dataset.py")
        print()
        print("  2. Run benchmarks:")
        print("     python benchmark_real_sharegpt.py")
        print("     python comprehensive_analysis.py")
        print("     python multi_config_benchmark.py")
    else:
        print("❌ SOME SCRIPTS HAVE ERRORS")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    main()
