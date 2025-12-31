"""
Pytest configuration for xfem-concrete tests.

Automatically adds src/ and repo root to sys.path so tests can import
xfem_clean and examples namespace without PYTHONPATH.
"""

import sys
import os

# Add repo root and src/ to path for module imports
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(repo_root, 'src')

# Add repo root first so 'examples' namespace is available
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Then add src/ for xfem_clean imports
if src_path not in sys.path:
    sys.path.insert(0, src_path)
