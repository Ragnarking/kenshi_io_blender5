"""PhysX 2.8.4 collision cooking through ctypes.

The bundled PhysX DLLs are plain C++ libraries without a Python binding, so this
module talks to them directly: the cooking library and the SDK are reached through
their exported factory functions and called through their virtual tables, and the
NxStream the SDK reads from or writes to is a ctypes object with its own table.
The vtable slots and descriptor layouts were taken from the original native module.
"""
import ctypes
import os
import sys
from ctypes import (CFUNCTYPE, POINTER, Structure, c_bool, c_char_p, c_double, c_float, c_int,
                    c_uint8, c_uint16, c_uint32, c_void_p, cast, sizeof)
from typing import List

_DIR = os.path.dirname(os.path.abspath(__file__))
NX_PHYSICS_SDK_VERSION = 0x02080400  # 2.8.4
_APP_ID = b'BE9042F0-ADC4-4b12-A93E-DB8A731FDBD5'

# NxCookingInterface slots
_COOK_INIT = 3
_COOK_TRIANGLE_MESH = 5
_COOK_CONVEX_MESH = 6
# NxPhysicsSDK slots
_SDK_CREATE_TRIANGLE_MESH = 0x40 // 8
_SDK_CREATE_CONVEX_MESH = 0x90 // 8
# NxTriangleMesh / NxConvexMesh slots
_MESH_GET_COUNT = 2
_MESH_GET_FORMAT = 3
_MESH_GET_BASE = 4
_MESH_GET_STRIDE = 5

NX_ARRAY_TRIANGLES = 0
NX_ARRAY_VERTICES = 1
NX_FORMAT_FLOAT = 1
NX_FORMAT_BYTE = 2
NX_FORMAT_SHORT = 3
NX_FORMAT_INT = 4
NX_CF_COMPUTE_CONVEX = 4
NX_NOT_HEIGHTFIELD = 0xFF


class NxConvexMeshDesc(Structure):
    _fields_ = [('numVertices', c_uint32), ('numTriangles', c_uint32),
                ('pointStrideBytes', c_uint32), ('triangleStrideBytes', c_uint32),
                ('points', c_void_p), ('triangles', c_void_p), ('flags', c_uint32),
                ('_reserved', c_uint8 * 28)]


class NxTriangleMeshDesc(Structure):
    _fields_ = [('numVertices', c_uint32), ('numTriangles', c_uint32),
                ('pointStrideBytes', c_uint32), ('triangleStrideBytes', c_uint32),
                ('points', c_void_p), ('triangles', c_void_p), ('flags', c_uint32),
                ('_base_padding', c_uint32),  # NxSimpleTriangleMesh is padded to 8 bytes
                ('materialIndexStride', c_uint32), ('materialIndices', c_void_p),
                ('heightFieldVerticalAxis', c_uint32), ('heightFieldVerticalExtent', c_float),
                ('pmap', c_void_p), ('convexEdgeThreshold', c_float),
                ('_reserved', c_uint8 * 28)]


# ---------------------------------------------------------------- NxStream

_this = c_void_p
_DTOR = CFUNCTYPE(c_void_p, _this, c_uint32)
_READ_U8 = CFUNCTYPE(c_uint8, _this)
_READ_U16 = CFUNCTYPE(c_uint16, _this)
_READ_U32 = CFUNCTYPE(c_uint32, _this)
_READ_F32 = CFUNCTYPE(c_float, _this)
_READ_F64 = CFUNCTYPE(c_double, _this)
_READ_BUF = CFUNCTYPE(c_void_p, _this, c_void_p, c_uint32)
_STORE_U8 = CFUNCTYPE(c_void_p, _this, c_uint8)
_STORE_U16 = CFUNCTYPE(c_void_p, _this, c_uint16)
_STORE_U32 = CFUNCTYPE(c_void_p, _this, c_uint32)
_STORE_F32 = CFUNCTYPE(c_void_p, _this, c_float)
_STORE_F64 = CFUNCTYPE(c_void_p, _this, c_double)
_STORE_BUF = CFUNCTYPE(c_void_p, _this, c_void_p, c_uint32)


class _StreamObject(Structure):
    _fields_ = [('vtable', c_void_p), ('id', c_uint32)]


class MemoryStream:
    """A C++ NxStream backed by a Python bytearray."""
    _streams = {}
    _next_id = 1

    def __init__(self, data: bytes = b''):
        self.buffer = bytearray(data)
        self.pos = 0
        self.id = MemoryStream._next_id
        MemoryStream._next_id += 1
        MemoryStream._streams[self.id] = self
        self.obj = _StreamObject(cast(_VTABLE, c_void_p), self.id)

    def close(self):
        MemoryStream._streams.pop(self.id, None)

    @property
    def ref(self):
        return ctypes.addressof(self.obj)

    def _read(self, n: int) -> bytes:
        b = bytes(self.buffer[self.pos:self.pos + n])
        self.pos += n
        return b.ljust(n, b'\0')

    @staticmethod
    def of(this) -> 'MemoryStream':
        return MemoryStream._streams[_StreamObject.from_address(this).id]


def _read_num(fmt):
    import struct
    size = struct.calcsize(fmt)
    return lambda this: struct.unpack('<' + fmt, MemoryStream.of(this)._read(size))[0]


def _store_num(fmt):
    import struct

    def store(this, value):
        MemoryStream.of(this).buffer += struct.pack('<' + fmt, value)
        return this
    return store


def _read_buffer(this, dest, size):
    ctypes.memmove(dest, MemoryStream.of(this)._read(size), size)
    return this


def _store_buffer(this, src, size):
    MemoryStream.of(this).buffer += ctypes.string_at(src, size)
    return this


_CALLBACKS = [
    _DTOR(lambda this, flags: this),
    _READ_U8(_read_num('B')), _READ_U16(_read_num('H')), _READ_U32(_read_num('I')),
    _READ_F32(_read_num('f')), _READ_F64(_read_num('d')), _READ_BUF(_read_buffer),
    _STORE_U8(_store_num('B')), _STORE_U16(_store_num('H')), _STORE_U32(_store_num('I')),
    _STORE_F32(_store_num('f')), _STORE_F64(_store_num('d')), _STORE_BUF(_store_buffer),
]
_VTABLE = (c_void_p * len(_CALLBACKS))(*[cast(cb, c_void_p) for cb in _CALLBACKS])


# ---------------------------------------------------------------- vtable calls

def _vcall(obj: int, slot: int, restype, *args):
    """args are (ctypes type, value) pairs."""
    vtable = c_void_p.from_address(obj).value
    fn_addr = c_void_p.from_address(vtable + slot * sizeof(c_void_p)).value
    proto = CFUNCTYPE(restype, c_void_p, *[t for t, _ in args])
    return proto(fn_addr)(obj, *[v for _, v in args])


class CollisionMesh:
    def __init__(self, points=None, triangles=None):
        self.points: List[List[float]] = points if points is not None else []
        self.triangles: List[int] = triangles if triangles is not None else []


class _PhysX:
    """Lazily loaded process wide PhysX SDK and cooking library, like the original module."""
    sdk = None
    cooking = None

    @classmethod
    def load(cls):
        if cls.cooking is not None:
            return cls
        if sys.platform != 'win32':
            raise RuntimeError('Kenshi PhysX cooking needs the bundled Windows PhysX libraries')
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(_DIR)
        cwd = os.getcwd()
        error = c_int(0)
        try:
            # the loader finds PhysXCore64 / PhysXCooking64 relative to the working directory
            os.chdir(_DIR)
            loader = ctypes.CDLL(os.path.join(_DIR, 'PhysXLoader64.dll'))
            loader.NxGetCookingLib.restype = c_void_p
            loader.NxGetCookingLib.argtypes = [c_uint32]
            loader.NxCreatePhysicsSDK.restype = c_void_p
            loader.NxCreatePhysicsSDK.argtypes = [c_uint32, c_void_p, c_void_p, c_void_p, c_void_p, c_char_p]
            desc = (c_uint32 * 16)(0x10000, 0x100, 0x800, 0, 3, 0x20, 0xFFFFFFFF)
            sdk = loader.NxCreatePhysicsSDK(NX_PHYSICS_SDK_VERSION, None, None, ctypes.addressof(desc),
                                            ctypes.addressof(error), _APP_ID)
            cooking = loader.NxGetCookingLib(NX_PHYSICS_SDK_VERSION)
        finally:
            os.chdir(cwd)
        if not sdk:
            raise RuntimeError(f'Failed to create the PhysX SDK (error {error.value})')
        if not cooking:
            raise RuntimeError('Failed to load the PhysX cooking library')
        _vcall(cooking, _COOK_INIT, c_bool, (c_void_p, None), (c_void_p, None))
        cls.loader, cls.sdk, cls.cooking = loader, sdk, cooking
        return cls


def _read_mesh(mesh: int) -> CollisionMesh:
    def info(array):
        count = _vcall(mesh, _MESH_GET_COUNT, c_uint32, (c_uint32, 0), (c_int, array))
        fmt = _vcall(mesh, _MESH_GET_FORMAT, c_int, (c_uint32, 0), (c_int, array))
        base = _vcall(mesh, _MESH_GET_BASE, c_void_p, (c_uint32, 0), (c_int, array))
        stride = _vcall(mesh, _MESH_GET_STRIDE, c_uint32, (c_uint32, 0), (c_int, array))
        return count, fmt, base, stride

    out = CollisionMesh()
    count, fmt, base, stride = info(NX_ARRAY_VERTICES)
    for i in range(count):
        p = (c_float * 3).from_address(base + i * stride)
        out.points.append([p[0], p[1], p[2]])
    count, fmt, base, stride = info(NX_ARRAY_TRIANGLES)
    index_type = {NX_FORMAT_BYTE: c_uint8, NX_FORMAT_SHORT: c_uint16, NX_FORMAT_INT: c_uint32}.get(fmt)
    if index_type is not None and base:
        for i in range(count):
            out.triangles.extend((index_type * 3).from_address(base + i * stride))
    return out


class KenshiPhysXSerializer:
    def __init__(self):
        self._px = _PhysX.load()

    def _cook(self, slot, desc) -> str:
        stream = MemoryStream()
        try:
            ok = _vcall(self._px.cooking, slot, c_bool, (c_void_p, ctypes.addressof(desc)), (c_void_p, stream.ref))
            return bytes(stream.buffer).hex().upper() if ok else ''
        finally:
            stream.close()

    def export_convex_hull(self, points) -> str:
        if len(points) < 3:
            return ''
        verts = (c_float * (3 * len(points)))(*[float(c) for p in points for c in p[:3]])
        desc = NxConvexMeshDesc(numVertices=len(points), pointStrideBytes=12,
                                points=ctypes.addressof(verts), flags=NX_CF_COMPUTE_CONVEX)
        return self._cook(_COOK_CONVEX_MESH, desc)

    def export_triangle_mesh(self, points, triangles) -> str:
        if len(points) < 3 or not triangles or len(triangles) % 3:
            return ''
        verts = (c_float * (3 * len(points)))(*[float(c) for p in points for c in p[:3]])
        tris = (c_uint32 * len(triangles))(*[int(i) for i in triangles])
        desc = NxTriangleMeshDesc(numVertices=len(points), numTriangles=len(triangles) // 3,
                                  pointStrideBytes=12, triangleStrideBytes=12,
                                  points=ctypes.addressof(verts), triangles=ctypes.addressof(tris),
                                  heightFieldVerticalAxis=NX_NOT_HEIGHTFIELD, convexEdgeThreshold=0.001)
        return self._cook(_COOK_TRIANGLE_MESH, desc)

    def _import(self, cooked_hex: str, create_slot) -> CollisionMesh:
        stream = MemoryStream(bytes.fromhex(''.join(cooked_hex.split())))
        try:
            mesh = _vcall(self._px.sdk, create_slot, c_void_p, (c_void_p, stream.ref))
        finally:
            stream.close()
        # like the original module the mesh is left to the SDK, which frees it on shutdown
        return _read_mesh(mesh) if mesh else CollisionMesh()

    def import_convex_hull(self, cooked_hex: str) -> CollisionMesh:
        return self._import(cooked_hex, _SDK_CREATE_CONVEX_MESH)

    def import_triangle_mesh(self, cooked_hex: str) -> CollisionMesh:
        return self._import(cooked_hex, _SDK_CREATE_TRIANGLE_MESH)
