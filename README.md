# Kenshi IO Tools for Blender 4.4+ / 5.x

A port of [Lucius64's kenshi_io_blender](https://github.com/Lucius64/kenshi_io_blender)
1.3.2 that runs on **Blender 5.1, 5.2 and later**. It imports and exports Kenshi
`.mesh` / `.skeleton` files with animations and shape keys, plus PhysX collision `.xml` files.

## Why this fork exists

Version 1.3.2 works up to Blender 5.0, but it relies on a closed-source native module,
`Kenshi_blender_tool.cp311-win_amd64.pyd`, built only for Python 3.11. Blender 5.1 moved
to Python 3.13, so installing the add-on there fails
([issue #4](https://github.com/Lucius64/kenshi_io_blender/issues/4)). The source of that
module is not public ([issue #2](https://github.com/Lucius64/kenshi_io_blender/issues/2)),
so it cannot be rebuilt.

This fork replaces the native module with a **pure Python package of the same interface**.
It no longer depends on a particular Python version.

| Module | Purpose |
|---|---|
| `kenshi_blender_tool/_mesh_file.py` | Ogre `.mesh` reader/writer (v1.41, v1.8, v1.10, v1.100) |
| `kenshi_blender_tool/_skeleton_file.py` | Ogre `.skeleton` reader/writer (v1.10, v1.80) |
| `kenshi_blender_tool/_mesh.py`, `_skeleton.py` | conversion to and from Blender: axes, UVs, colours, weights, poses, animations, helper bones |
| `kenshi_blender_tool/_physx.py` | collision cooking through the bundled PhysX 2.8.4 DLLs via `ctypes` |
| `kenshi_blender_tool/_serializer.py` | `KenshiObjectSerializer` |

The add-on's Python files (operators, importers, exporters) are unchanged from 1.3.2,
except one Blender 5 API fix in `ogre_exporter.py`. `OgreMain.dll` is no longer needed.

## Installation

1. Download `kenshi_io_tools-<version>.zip` from Releases.
2. In Blender: *Edit → Preferences → Get Extensions → ▾ → Install from Disk…* and pick the zip.
3. The operators appear under *File → Import / Export* (Kenshi mesh, skeleton, collision).

Disable or remove the original *Kenshi IO Tools* extension first if you have it installed.

Meshes and skeletons work on every platform. Collision import/export needs the Windows
PhysX DLLs that ship with the add-on, as in the original.

## How it was verified

The original cp310 native module (from the 3.6 release) served as a reference oracle. It ran
under Python 3.10 with numpy 1.x, because with numpy 2 it returns broken 1D arrays.

- **Reading:** 396 meshes and 363 skeletons from Kenshi's game data give the same data as the original.
- **Mesh writing:** 42 generated files are byte-identical across all versions, tangent formats and optimisation settings.
- **PhysX:** cooked data is byte-identical, and 161 collision shapes from the game read back identically.
- **Skeleton writing:** matches within float32 rounding, including Ogre's duplicate-keyframe optimisation.
- **Blender 5.2.3:** `tests/blender_smoke.py` passes all 18 checks, both from source and as an installed extension. It covers a mesh with skeleton and 9 animations, export and reimport, 36 shape keys, baked skeleton export and collision round trip.

Deliberate differences from the original:

- Scale keyframes now use the same axis mapping as rotations. The original permuted scale axes inconsistently between import and its two export paths. Vanilla Kenshi animations contain no scale keys.
- One vanilla track takes the shorter rotation path where the original picked the opposite quaternion sign (`male_skeleton`, *arrow dodge spin 2*, *Bip01 Prop1*).
- `calc_animation_fps`, used only by *Round frames*, differs for 1 of 29 animated skeletons (4.0 vs 4.5).

Only Blender 5.2.3 has been tested; the manifest allows 4.4 and later.

## Building

```
blender --command extension build --source-dir addon --output-dir dist
```

## Tests

Run from the repository root. Set `KENSHI_DATA` to Kenshi's `data` folder. The oracle steps
also need `KENSHI_ORACLE`: a folder holding `Kenshi_blender_tool.cp310-win_amd64.pyd` and its DLLs
from the original 3.6 release, used with Python 3.10 + numpy 1.x.

```
# reference dumps (Python 3.10 + numpy 1.x)
python tests/pick_meshes.py > meshlist.txt
python tests/dump_oracle_mesh.py out/oracle_mesh meshlist.txt
python tests/dump_oracle_skel.py out/oracle_skel
python tests/oracle_export.py out/oracle_export
python tests/oracle_skel_export.py out/oracle_skel_export
python tests/oracle_physx.py out/oracle_physx.pkl

# comparisons (any Python 3.10+ with numpy)
python tests/compare_mesh.py out/oracle_mesh
python tests/compare_skel.py out/oracle_skel
python tests/compare_export.py out/oracle_export
python tests/compare_skel_export.py out/oracle_skel_export
python tests/compare_physx.py out/oracle_physx.pkl

# inside Blender, with the add-on at <dir>/addons/kenshi_io_tools
BLENDER_USER_SCRIPTS=<dir> blender -b --factory-startup --python tests/blender_smoke.py -- <dir> out/blender
```

## Credits and license

- Original add-on and native module design: **Lucius64**, [kenshi_io_blender](https://github.com/Lucius64/kenshi_io_blender), GPL-3.0-or-later.
- The pure Python reimplementation of `kenshi_blender_tool` is written from the documented Ogre file formats and the module's observable behaviour. The PhysX vtable slots and descriptor layouts needed for compatibility come from the original binary.
- Ogre file format: OGRE (MIT). PhysX 2.8.4 runtime DLLs: NVIDIA, redistributed as in the original add-on (see `addon/NOTICE.md`).

This project is licensed under GPL-3.0-or-later; see `addon/LICENSE`.
