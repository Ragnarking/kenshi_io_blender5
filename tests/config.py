"""Paths used by the tests, overridable with environment variables.

KENSHI_DATA    Kenshi's data folder (game files used as test input)
KENSHI_ORACLE  folder holding the original native module, Kenshi_blender_tool.cp310-win_amd64.pyd
               and its DLLs (the 3.6 release of kenshi_io_blender), used as the reference
"""
import os
import tempfile

KENSHI_DATA = os.environ.get('KENSHI_DATA', 'C:/Program Files (x86)/Steam/steamapps/common/Kenshi/data').replace('\\', '/')
ORACLE_DIR = os.environ.get('KENSHI_ORACLE', os.path.join(
    os.path.dirname(__file__), '..', 'sources', 'orig_kenshi_io_blender', '3.6', 'Kenshi_io_tools', 'kenshi_blender_tool'))
LOG = os.path.join(tempfile.gettempdir(), 'kenshi_io_oracle.log')
