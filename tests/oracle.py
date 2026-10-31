"""Load the original cp310 Kenshi_blender_tool (Python 3.10 only) for differential tests."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from config import ORACLE_DIR  # noqa: E402
_DIR = os.path.abspath(ORACLE_DIR)
os.add_dll_directory(_DIR)
sys.path.insert(0, _DIR)
import Kenshi_blender_tool as K  # noqa
