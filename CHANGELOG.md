# Changelog

## 1.4.0

Fork of kenshi_io_blender 1.3.2.

- Replaced the closed-source `Kenshi_blender_tool` native module, built for CPython 3.11 only,
  with a pure Python package of the same interface. The add-on now installs and runs on
  Blender 5.1+ (Python 3.13).
- PhysX collision cooking calls the bundled PhysX 2.8.4 DLLs directly through `ctypes`.
- Fixed baked skeleton export on Blender 5 (`slots.get(..., default=...)` no longer accepts the keyword).
- Scale keyframes now use the same axis mapping as rotations on import and export.
- `OgreMain.dll` is no longer bundled.
