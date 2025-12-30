"""
Pytest configuration for xfem-concrete tests.

Automatically adds src/ to sys.path so tests can import xfem_clean without PYTHONPATH.
"""

import sys
import os

# Add src/ to path for module imports
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(repo_root, 'src')

if src_path not in sys.path:
    sys.path.insert(0, src_path)
