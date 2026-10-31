"""Binary Ogre .mesh reader/writer (formats v1.41, v1.8, v1.10 and v1.100).

This is a pure Python replacement for the Ogre MeshSerializer that the
original native module used, so the add-on no longer depends on a CPython
version specific binary.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._stream import Reader, Writer

M_HEADER = 0x1000
M_MESH = 0x3000
M_SUBMESH = 0x4000
M_SUBMESH_OPERATION = 0x4010
M_SUBMESH_BONE_ASSIGNMENT = 0x4100
M_SUBMESH_TEXTURE_ALIAS = 0x4200
M_GEOMETRY = 0x5000
M_GEOMETRY_VERTEX_DECLARATION = 0x5100
M_GEOMETRY_VERTEX_ELEMENT = 0x5110
M_GEOMETRY_VERTEX_BUFFER = 0x5200
M_GEOMETRY_VERTEX_BUFFER_DATA = 0x5210
M_MESH_SKELETON_LINK = 0x6000
M_MESH_BONE_ASSIGNMENT = 0x7000
M_MESH_LOD_LEVEL = 0x8000
M_MESH_BOUNDS = 0x9000
M_SUBMESH_NAME_TABLE = 0xA000
M_SUBMESH_NAME_TABLE_ELEMENT = 0xA100
M_EDGE_LISTS = 0xB000
M_POSES = 0xC000
M_POSE = 0xC100
M_POSE_VERTEX = 0xC111
M_ANIMATIONS = 0xD000
M_TABLE_EXTREMES = 0xE000

VERSION_1_100 = '[MeshSerializer_v1.100]'
VERSION_1_10 = '[MeshSerializer_v1.10]'
VERSION_1_8 = '[MeshSerializer_v1.8]'
VERSION_1_41 = '[MeshSerializer_v1.41]'
VERSION_1_4 = '[MeshSerializer_v1.40]'

# Vertex element semantics
VES_POSITION = 1
VES_BLEND_WEIGHTS = 2
VES_BLEND_INDICES = 3
VES_NORMAL = 4
VES_DIFFUSE = 5
VES_SPECULAR = 6
VES_TEXTURE_COORDINATES = 7
VES_BINORMAL = 8
VES_TANGENT = 9

# Vertex element types: (numpy dtype, component count)
VET_FLOAT1, VET_FLOAT2, VET_FLOAT3, VET_FLOAT4 = 0, 1, 2, 3
VET_COLOUR = 4
VET_UBYTE4 = 9
VET_COLOUR_ARGB = 10
VET_COLOUR_ABGR = 11
VET_FORMATS = {
    0: ('<f4', 1), 1: ('<f4', 2), 2: ('<f4', 3), 3: ('<f4', 4),
    4: ('u1', 4),
    5: ('<i2', 1), 6: ('<i2', 2), 7: ('<i2', 3), 8: ('<i2', 4),
    9: ('u1', 4), 10: ('u1', 4), 11: ('u1', 4),
    12: ('<f8', 1), 13: ('<f8', 2), 14: ('<f8', 3), 15: ('<f8', 4),
    16: ('<u2', 1), 17: ('<u2', 2), 18: ('<u2', 3), 19: ('<u2', 4),
    20: ('<i4', 1), 21: ('<i4', 2), 22: ('<i4', 3), 23: ('<i4', 4),
    24: ('<u4', 1), 25: ('<u4', 2), 26: ('<u4', 3), 27: ('<u4', 4),
}


def element_size(vtype: int) -> int:
    dt, n = VET_FORMATS[vtype]
    return np.dtype(dt).itemsize * n


@dataclass
class VertexElement:
    source: int
    type: int
    semantic: int
    offset: int
    index: int


@dataclass
class VertexData:
    vertex_count: int = 0
    elements: List[VertexElement] = field(default_factory=list)
    buffers: Dict[int, Tuple[int, bytes]] = field(default_factory=dict)  # bind -> (vertex size, data)

    def find(self, semantic: int, index: int = 0) -> Optional[VertexElement]:
        for e in self.elements:
            if e.semantic == semantic and e.index == index:
                return e
        return None

    def read_element(self, elem: VertexElement) -> np.ndarray:
        """Returns a (vertex_count, components) array of the element."""
        vsize, data = self.buffers[elem.source]
        dt, n = VET_FORMATS[elem.type]
        size = np.dtype(dt).itemsize * n
        rows = np.frombuffer(data, dtype=np.uint8, count=vsize * self.vertex_count).reshape(-1, vsize)
        raw = np.ascontiguousarray(rows[:, elem.offset:elem.offset + size])
        return raw.view(np.dtype(dt)).reshape(self.vertex_count, n).astype(np.dtype(dt).newbyteorder('='))


@dataclass
class Pose:
    name: bytes
    target: int  # 0 shared geometry, 1+ submesh index + 1
    include_normals: bool
    indices: np.ndarray
    offsets: np.ndarray
    normals: Optional[np.ndarray] = None


@dataclass
class SubMeshFile:
    material: bytes = b''
    use_shared_vertices: bool = False
    indices: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint32))
    use_32bit_indexes: bool = False
    vertex_data: Optional[VertexData] = None
    operation_type: int = 4  # OT_TRIANGLE_LIST
    bone_assignments: List[Tuple[int, int, float]] = field(default_factory=list)
    texture_aliases: List[Tuple[bytes, bytes]] = field(default_factory=list)


@dataclass
class MeshFile:
    version: str = VERSION_1_100
    skeletally_animated: bool = False
    shared_vertex_data: Optional[VertexData] = None
    submeshes: List[SubMeshFile] = field(default_factory=list)
    skeleton_name: bytes = b''
    bone_assignments: List[Tuple[int, int, float]] = field(default_factory=list)
    bounds: Tuple[float, ...] = (0, 0, 0, 0, 0, 0, 0)
    submesh_names: Dict[int, bytes] = field(default_factory=dict)
    poses: List[Pose] = field(default_factory=list)


# ---------------------------------------------------------------- reading
#
# Chunks are parsed sequentially like Ogre does. Many v1.8 / v1.41 files were
# written by exporters that stored wrong chunk lengths, so lengths are only
# trusted for skipping chunks in files written by Ogre itself (v1.10+).

M_MESH_LOD_USAGE = 0x8100
M_MESH_LOD_MANUAL = 0x8110
M_MESH_LOD_GENERATED = 0x8120
M_EDGE_LIST_LOD = 0xB100
M_EDGE_GROUP = 0xB110
M_ANIMATION = 0xD100
M_ANIMATION_BASEINFO = 0xD105
M_ANIMATION_TRACK = 0xD110
M_ANIMATION_MORPH_KEYFRAME = 0xD111
M_ANIMATION_POSE_KEYFRAME = 0xD112
M_ANIMATION_POSE_REF = 0xD113

_MESH_CHUNKS = {M_GEOMETRY, M_SUBMESH, M_MESH_SKELETON_LINK, M_MESH_BONE_ASSIGNMENT,
                M_MESH_LOD_LEVEL, M_MESH_BOUNDS, M_SUBMESH_NAME_TABLE, M_EDGE_LISTS,
                M_POSES, M_ANIMATIONS, M_TABLE_EXTREMES}


class _MeshReader:
    def __init__(self, data: bytes):
        self.r = Reader(data)

    def next_chunk(self, allowed):
        """Reads a chunk header if the next chunk id is one of `allowed`."""
        r = self.r
        if r.pos + 6 > len(r.data) or r.peek_chunk_id() not in allowed:
            return None
        return r.chunk()

    def read(self) -> MeshFile:
        r = self.r
        if r.u16() != M_HEADER:
            raise ValueError('File header not found, not an Ogre mesh')
        version = r.string().decode('ascii', errors='replace')
        if version not in (VERSION_1_100, VERSION_1_10, VERSION_1_8, VERSION_1_41, VERSION_1_4):
            raise ValueError(f'Unsupported mesh version {version}')
        self.mesh = MeshFile(version=version)
        self.modern = version in (VERSION_1_100, VERSION_1_10)
        self.has_normals_flag = version not in (VERSION_1_41, VERSION_1_4)
        while self.next_chunk({M_MESH}):
            self.read_mesh()
        return self.mesh

    def read_mesh(self):
        r, mesh = self.r, self.mesh
        mesh.skeletally_animated = r.bool()
        while True:
            c = self.next_chunk(_MESH_CHUNKS)
            if c is None:
                return
            cid, start, end = c
            if cid == M_GEOMETRY:
                mesh.shared_vertex_data = self.read_geometry()
            elif cid == M_SUBMESH:
                mesh.submeshes.append(self.read_submesh())
            elif cid == M_MESH_SKELETON_LINK:
                mesh.skeleton_name = r.string()
            elif cid == M_MESH_BONE_ASSIGNMENT:
                mesh.bone_assignments.append(r.unpack('IHf'))
            elif cid == M_MESH_LOD_LEVEL:
                self.skip_lod(end)
            elif cid == M_MESH_BOUNDS:
                mesh.bounds = r.floats(7)
            elif cid == M_SUBMESH_NAME_TABLE:
                while self.next_chunk({M_SUBMESH_NAME_TABLE_ELEMENT}):
                    index = r.u16()
                    mesh.submesh_names[index] = r.string()
            elif cid == M_EDGE_LISTS:
                self.skip_edge_lists()
            elif cid == M_POSES:
                while self.next_chunk({M_POSE}):
                    mesh.poses.append(self.read_pose())
            elif cid == M_ANIMATIONS:
                self.skip_animations()
            elif cid == M_TABLE_EXTREMES:
                r.pos = end  # Ogre itself relies on the length here

    def read_geometry(self) -> VertexData:
        r = self.r
        vd = VertexData(vertex_count=r.u32())
        while True:
            c = self.next_chunk({M_GEOMETRY_VERTEX_DECLARATION, M_GEOMETRY_VERTEX_BUFFER})
            if c is None:
                return vd
            if c[0] == M_GEOMETRY_VERTEX_DECLARATION:
                while self.next_chunk({M_GEOMETRY_VERTEX_ELEMENT}):
                    vd.elements.append(VertexElement(*r.unpack('5H')))
            else:
                bind, vsize = r.unpack('2H')
                if not self.next_chunk({M_GEOMETRY_VERTEX_BUFFER_DATA}):
                    raise ValueError('Missing vertex buffer data')
                vd.buffers[bind] = (vsize, r.bytes(vsize * vd.vertex_count))

    def read_submesh(self) -> SubMeshFile:
        r = self.r
        sm = SubMeshFile()
        sm.material = r.string()
        sm.use_shared_vertices = r.bool()
        index_count = r.u32()
        sm.use_32bit_indexes = r.bool()
        if index_count > 0:
            sm.indices = r.array('u4' if sm.use_32bit_indexes else 'u2', index_count).astype(np.uint32)
        if not sm.use_shared_vertices:
            if not self.next_chunk({M_GEOMETRY}):
                raise ValueError('Missing geometry data in submesh')
            sm.vertex_data = self.read_geometry()
        while True:
            c = self.next_chunk({M_SUBMESH_OPERATION, M_SUBMESH_BONE_ASSIGNMENT, M_SUBMESH_TEXTURE_ALIAS})
            if c is None:
                return sm
            if c[0] == M_SUBMESH_OPERATION:
                sm.operation_type = r.u16()
            elif c[0] == M_SUBMESH_BONE_ASSIGNMENT:
                sm.bone_assignments.append(r.unpack('IHf'))
            else:
                sm.texture_aliases.append((r.string(), r.string()))

    def read_pose(self) -> Pose:
        r = self.r
        name = r.string()
        target = r.u16()
        include_normals = r.bool() if self.has_normals_flag else False
        n = 6 if include_normals else 3
        indices, values = [], []
        while self.next_chunk({M_POSE_VERTEX}):
            indices.append(r.u32())
            values.append(r.floats(n))
        values = np.array(values, dtype=np.float32).reshape(-1, n)
        return Pose(name, target, include_normals, np.array(indices, dtype=np.uint32),
                    values[:, :3].copy(), values[:, 3:].copy() if include_normals else None)

    # --- chunks the add-on does not use, parsed only to stay in sync

    def skip_lod(self, end):
        r = self.r
        if self.modern:
            r.pos = end
            return
        r.string()  # LOD strategy name, present since v1.41
        num_levels = r.u16()
        manual = r.bool()
        for _ in range(1, num_levels):
            if not self.next_chunk({M_MESH_LOD_USAGE}):
                raise ValueError('Missing LOD usage')
            r.f32()
            if manual:
                if not self.next_chunk({M_MESH_LOD_MANUAL}):
                    raise ValueError('Missing manual LOD')
                r.string()
            else:
                for _sm in self.mesh.submeshes:
                    if not self.next_chunk({M_MESH_LOD_GENERATED}):
                        raise ValueError('Missing generated LOD')
                    count = r.u32()
                    is32 = r.bool()
                    r.bytes(count * (4 if is32 else 2))

    def skip_edge_lists(self):
        r = self.r
        while self.next_chunk({M_EDGE_LIST_LOD}):
            r.u16()
            if r.bool():  # manual LOD, no data
                continue
            r.bool()  # is closed
            num_triangles, num_groups = r.unpack('2I')
            r.bytes(num_triangles * 48)
            for _ in range(num_groups):
                if not self.next_chunk({M_EDGE_GROUP}):
                    raise ValueError('Missing edge group')
                num_edges = r.unpack('4I')[3]
                r.bytes(num_edges * 25)

    def skip_animations(self):
        r = self.r
        while self.next_chunk({M_ANIMATION}):
            r.string()
            r.f32()
            if self.next_chunk({M_ANIMATION_BASEINFO}):
                r.string()
                r.f32()
            while self.next_chunk({M_ANIMATION_TRACK}):
                _anim_type, target = r.unpack('2H')
                vd = self.mesh.shared_vertex_data if target == 0 else \
                    (self.mesh.submeshes[target - 1].vertex_data or self.mesh.shared_vertex_data)
                while True:
                    c = self.next_chunk({M_ANIMATION_MORPH_KEYFRAME, M_ANIMATION_POSE_KEYFRAME})
                    if c is None:
                        break
                    r.f32()
                    if c[0] == M_ANIMATION_MORPH_KEYFRAME:
                        normals = r.bool() if self.has_normals_flag else False
                        r.bytes(vd.vertex_count * (24 if normals else 12))
                    else:
                        while self.next_chunk({M_ANIMATION_POSE_REF}):
                            r.unpack('Hf')


def read_mesh(data: bytes) -> MeshFile:
    return _MeshReader(data).read()


# ---------------------------------------------------------------- writing

def write_mesh(mesh: MeshFile) -> bytes:
    """Writes the chunks Ogre's MeshSerializer emits for a mesh built by the exporter."""
    version = mesh.version
    pose_normals_flag = version != VERSION_1_4
    w = Writer()
    w.u16(M_HEADER)
    w.string(version)
    w.begin_chunk(M_MESH)
    w.bool(mesh.skeletally_animated)
    if mesh.shared_vertex_data is not None:
        _write_geometry(w, mesh.shared_vertex_data)
    for sm in mesh.submeshes:
        _write_submesh(w, sm)
    if mesh.skeleton_name:
        w.begin_chunk(M_MESH_SKELETON_LINK)
        w.string(mesh.skeleton_name)
        w.end_chunk()
    for ba in mesh.bone_assignments:
        w.begin_chunk(M_MESH_BONE_ASSIGNMENT)
        w.pack('IHf', *ba)
        w.end_chunk()
    w.begin_chunk(M_MESH_BOUNDS)
    w.floats(*mesh.bounds)
    w.end_chunk()
    if mesh.submesh_names:
        w.begin_chunk(M_SUBMESH_NAME_TABLE)
        for index in sorted(mesh.submesh_names):
            w.begin_chunk(M_SUBMESH_NAME_TABLE_ELEMENT)
            w.u16(index)
            w.string(mesh.submesh_names[index])
            w.end_chunk()
        w.end_chunk()
    if mesh.poses:
        w.begin_chunk(M_POSES)
        for pose in mesh.poses:
            w.begin_chunk(M_POSE)
            w.string(pose.name)
            w.u16(pose.target)
            if pose_normals_flag:
                w.bool(pose.include_normals)
            for i, vi in enumerate(pose.indices):
                w.begin_chunk(M_POSE_VERTEX)
                w.u32(int(vi))
                w.floats(*pose.offsets[i])
                if pose.include_normals:
                    w.floats(*pose.normals[i])
                w.end_chunk()
            w.end_chunk()
        w.end_chunk()
    w.end_chunk()
    return w.getvalue()


def _write_submesh(w: Writer, sm: SubMeshFile):
    w.begin_chunk(M_SUBMESH)
    w.string(sm.material)
    w.bool(sm.use_shared_vertices)
    w.u32(len(sm.indices))
    w.bool(sm.use_32bit_indexes)
    if len(sm.indices):
        w.array(sm.indices, 'u4' if sm.use_32bit_indexes else 'u2')
    if not sm.use_shared_vertices:
        _write_geometry(w, sm.vertex_data)
    w.begin_chunk(M_SUBMESH_OPERATION)
    w.u16(sm.operation_type)
    w.end_chunk()
    for alias, texture in sm.texture_aliases:
        w.begin_chunk(M_SUBMESH_TEXTURE_ALIAS)
        w.string(alias)
        w.string(texture)
        w.end_chunk()
    for ba in sm.bone_assignments:
        w.begin_chunk(M_SUBMESH_BONE_ASSIGNMENT)
        w.pack('IHf', *ba)
        w.end_chunk()
    w.end_chunk()


def _write_geometry(w: Writer, vd: VertexData):
    w.begin_chunk(M_GEOMETRY)
    w.u32(vd.vertex_count)
    w.begin_chunk(M_GEOMETRY_VERTEX_DECLARATION)
    for e in vd.elements:
        w.begin_chunk(M_GEOMETRY_VERTEX_ELEMENT)
        w.pack('5H', e.source, e.type, e.semantic, e.offset, e.index)
        w.end_chunk()
    w.end_chunk()
    for bind in sorted(vd.buffers):
        vsize, data = vd.buffers[bind]
        w.begin_chunk(M_GEOMETRY_VERTEX_BUFFER)
        w.pack('2H', bind, vsize)
        w.begin_chunk(M_GEOMETRY_VERTEX_BUFFER_DATA)
        w.raw(data)
        w.end_chunk()
        w.end_chunk()
    w.end_chunk()


def build_vertex_data(columns) -> VertexData:
    """columns: list of (semantic, vertex element type, index, (n, k) array) in declaration order,
    interleaved into a single buffer bound to source 0."""
    count = len(columns[0][3]) if columns else 0
    elements, fields, offset = [], [], 0
    for i, (semantic, vtype, index, values) in enumerate(columns):
        dt, n = VET_FORMATS[vtype]
        elements.append(VertexElement(0, vtype, semantic, offset, index))
        fields.append((f'f{i}', np.dtype(dt), (n,)))
        offset += element_size(vtype)
    rec = np.zeros(count, dtype=np.dtype(fields))
    for i, (_, vtype, _, values) in enumerate(columns):
        rec[f'f{i}'] = values.reshape(count, -1)
    return VertexData(count, elements, {0: (offset, rec.tobytes())})
