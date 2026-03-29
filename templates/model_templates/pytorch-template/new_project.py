#!/usr/bin/env python3
"""
Initialize a new project by copying the pytorch-template directory using the
unified tools.py interface.

Usage:
  python new_project.py <target_dir>
"""
import sys
from pathlib import Path
import subprocess
import shutil

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent / 'tools.py'


def main():
    if len(sys.argv) != 2:
        print('Usage: python new_project.py <target_dir>')
        sys.exit(1)
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)

    # Preferred: delegate to tools.py for consistent behavior
    if TOOLS.exists():
        try:
            subprocess.check_call([
                'python', str(TOOLS), 'fetch', 'pytorch_template', str(target)
            ])
            return
        except Exception as e:
            print(f'[warn] tools.py fetch failed: {e}; falling back to direct copy')

    # Fallback: direct copy of current directory (excluding meta files)
    src = HERE
    ignore = shutil.ignore_patterns(
        '.git', 'data', 'saved', '__pycache__', '.flake8', 'README.md', 'LICENSE', 'new_project.py'
    )
    shutil.copytree(src, target / src.name, dirs_exist_ok=True, ignore=ignore)
    print('New project initialized at', (target / src.name).resolve())


if __name__ == '__main__':
    main()
