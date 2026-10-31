"""MeshData / SubMeshData with the interface of the original native module.

Blender coordinates are (x, -z, y) of Ogre coordinates and texture V is flipped.
"""
from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np

from . import _mesh_file as MF


class OperationType(IntEnum):
    triangle_list = 4


class MeshVersion(IntEnum):
    V_Latest = 0
    V_1_10 = 1
    V_1_8 = 2
    V_1_7 = 3
    V_1_4 = 4
    V_1_0 = 5
    V_Legacy = 6


def to_blender(v: np.ndarray) -> np.ndarray:
    """(n, 3) Ogre -> Blender"""
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)


def to_ogre(v: np.ndarray) -> np.ndarray:
    """(n, 3) Blender -> Ogre"""
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)


def _decode(b: bytes) -> str:
    return b.decode('utf-8', errors='replace')


class BoneAssignmentData:
    def __init__(self, vertex_index: int, bone_index: int, weight: float):
        self.vertex_index = int(vertex_index)
        self.bone_index = int(bone_index)
        self.weight = float(weight)

    def __repr__(self):
        return f'BoneAssignmentData({self.vertex_index}, {self.bone_index}, {self.weight})'


class VertexGroupData:
    def __init__(self, name: str = '', group=None):
        self._name = name
        self._group = group if group is not None else []

    @property
    def name(self) -> str:
        return self._name

    @property
    def group(self):
        return self._group


class GeometryData:
    def __init__(self, vertex_data: Optional[MF.VertexData] = None, shared: bool = False):
        self._vd = vertex_data
        self._shared = shared

    def _has(self, semantic):
        return self._vd is not None and self._vd.find(semantic) is not None

    @property
    def vertex_count(self) -> int:
        return self._vd.vertex_count if self._vd else 0

    @property
    def has_shared_geometry(self) -> bool:
        return self._shared

    @property
    def has_positions(self) -> bool:
        return self._has(MF.VES_POSITION)

    @property
    def has_normals(self) -> bool:
        return self._has(MF.VES_NORMAL)

    @property
    def has_colors(self) -> bool:
        return self._has(MF.VES_DIFFUSE)

    @property
    def has_texture_coord(self) -> bool:
        return self._has(MF.VES_TEXTURE_COORDINATES)

    @property
    def texcoords_size(self) -> int:
        if self._vd is None:
            return 0
        return sum(1 for e in self._vd.elements if e.semantic == MF.VES_TEXTURE_COORDINATES)

    @property
    def has_tangents(self) -> bool:
        return self._has(MF.VES_TANGENT)

    @property
    def tangent_dimensions(self) -> int:
        e = self._vd.find(MF.VES_TANGENT) if self._vd else None
        if e is None:
            return 3
        return MF.VET_FORMATS[e.type][1]

    @property
    def has_binormals(self) -> bool:
        return self._has(MF.VES_BINORMAL)


class SubMeshData:
    def __init__(self):
        self.index = 0
        self.submesh_name = ''
        self.material = ''
        self.use_32bit_indexes = False
        self.use_shared_vertices = False
        self.operation_type = OperationType.triangle_list
        self.boneassignments: List[BoneAssignmentData] = []
        self.faces: List[int] = []
        self.face_count = 0
        self._name_bytes: Optional[bytes] = None
        self._material_bytes: Optional[bytes] = None
        self._vd: Optional[MF.VertexData] = None
        self._indices = np.zeros(0, dtype=np.int64)
        self._poses: List[MF.Pose] = []
        self._bone_map: Dict[int, str] = {}

    # ---------------------------------------------------------- export

    def set_vertex(self, nd_vert_indices, nd_loop_indices, nd_positions, nd_normals, nd_tangents,
                   nd_bitangent_signs, nd_bitangents, nd_texcoords, nd_colors, nd_alphas,
                   tangent_dimensions: int = 3, optimize: bool = False) -> np.ndarray:
        """Builds Ogre vertices from Blender face corners. An array without one entry per corner
        means that attribute is not exported. Returns the Blender vertex index of every Ogre vertex."""
        vert = np.asarray(nd_vert_indices, dtype=np.int64)
        loops = np.asarray(nd_loop_indices, dtype=np.int64)
        count = len(vert)

        def per_loop(a, k):
            a = np.asarray(a, dtype=np.float32)
            return a.reshape(-1, k)[loops] if count and a.size == count * k else None

        positions = to_ogre(np.asarray(nd_positions, dtype=np.float32).reshape(-1, 3)[vert])
        columns = [(MF.VES_POSITION, MF.VET_FLOAT3, positions)]
        normals = per_loop(nd_normals, 3)
        if normals is not None:
            columns.append((MF.VES_NORMAL, MF.VET_FLOAT3, to_ogre(normals)))
        tangents = per_loop(nd_tangents, 3)
        if tangents is not None:
            tangents = to_ogre(tangents)
            if tangent_dimensions == 4:
                signs = per_loop(nd_bitangent_signs, 1)
                signs = signs if signs is not None else np.ones((count, 1), np.float32)
                columns.append((MF.VES_TANGENT, MF.VET_FLOAT4, np.c_[tangents, signs]))
            else:
                columns.append((MF.VES_TANGENT, MF.VET_FLOAT3, tangents))
        bitangents = per_loop(nd_bitangents, 3)
        if bitangents is not None:
            # the original module stored the negated bitangent, the add-on's tangent formats rely on it
            columns.append((MF.VES_BINORMAL, MF.VET_FLOAT3, -to_ogre(bitangents)))
        colors = per_loop(nd_colors, 4)
        if colors is not None:
            alphas = per_loop(nd_alphas, 4)
            a = alphas[:, 0] if alphas is not None else np.ones(count, np.float32)
            # packed ARGB, converted like Ogre::ColourValue::getAsARGB (truncation)
            argb = np.stack([colors[:, 2], colors[:, 1], colors[:, 0], a], axis=1)
            columns.append((MF.VES_DIFFUSE, MF.VET_COLOUR_ARGB,
                            (np.clip(argb, 0.0, 1.0) * 255.0).astype(np.uint8)))
        texcoords = per_loop(nd_texcoords, 2)
        if texcoords is not None:
            columns.append((MF.VES_TEXTURE_COORDINATES, MF.VET_FLOAT2,
                            np.c_[texcoords[:, 0], 1.0 - texcoords[:, 1]]))
        columns = [(sem, vt, np.ascontiguousarray(v, dtype=np.uint8 if vt == MF.VET_COLOUR_ARGB else np.float32))
                   for sem, vt, v in columns]

        if optimize:
            # Merge corners of the same Blender vertex whose exported attributes are identical
            keyed = np.ascontiguousarray(np.concatenate(
                [vert.reshape(count, 1).view(np.uint8).reshape(count, -1)] +
                [v.reshape(count, -1).view(np.uint8) for _, _, v in columns], axis=1))
            keys = keyed.view(np.dtype((np.void, keyed.shape[1]))).ravel()
            _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
            order = np.argsort(first)  # unique keys in first occurrence order
            remap = np.empty(len(order), dtype=np.int64)
            remap[order] = np.arange(len(order))
            indices = remap[inverse.ravel()]
            picked = first[order]
            print(f'loop:{count} vertex:{len(picked)}')
            print(f'reduce {count - len(picked)} vertices.')
        else:
            indices = np.arange(count, dtype=np.int64)
            picked = indices
            print(f'loop:{count} vertex:{count}')

        self._vd = MF.build_vertex_data([(sem, vt, 0, v[picked]) for sem, vt, v in columns])
        self._indices = indices
        self.faces = indices.tolist()
        self.face_count = count // 3
        self.use_32bit_indexes = len(picked) > 65535
        self.use_shared_vertices = False
        self._blender_vertex = vert[picked].astype(np.int32)
        self._poses = []
        return self._blender_vertex.copy()

    def append_shapekey(self, name: str, nd_offsets, nd_indices):
        """nd_offsets: (Blender vertex count, 3) offsets, nd_indices: the result of set_vertex."""
        offsets = to_ogre(np.asarray(nd_offsets, dtype=np.float32).reshape(-1, 3)[np.asarray(nd_indices)])
        used = np.nonzero(np.any(offsets != 0, axis=1))[0]
        self._poses.append(MF.Pose(name.encode('utf-8'), self.index + 1, False, used.astype(np.uint32),
                                   offsets[used].astype(np.float32)))

    def set_bone_assignments(self, assignments, nd_indices):
        by_vertex: Dict[int, list] = {}
        for ba in assignments:
            by_vertex.setdefault(ba.vertex_index, []).append(ba)
        out = []
        for ogre_index, blender_index in enumerate(np.asarray(nd_indices).tolist()):
            for ba in by_vertex.get(blender_index, ()):
                out.append(BoneAssignmentData(ogre_index, ba.bone_index, ba.weight))
        self.boneassignments = out

    def _to_file(self) -> MF.SubMeshFile:
        vertex_count = self._vd.vertex_count if self._vd is not None else 0
        return MF.SubMeshFile(
            material=self.encorded_material,
            use_shared_vertices=False,
            indices=np.asarray(self._indices, dtype=np.uint32),
            use_32bit_indexes=self.use_32bit_indexes or vertex_count > 65535,
            vertex_data=self._vd,
            operation_type=int(self.operation_type),
            bone_assignments=[(b.vertex_index, b.bone_index, b.weight) for b in self.boneassignments])

    # ---------------------------------------------------------- import

    @classmethod
    def _from_file(cls, index: int, sm: MF.SubMeshFile, mesh: MF.MeshFile) -> 'SubMeshData':
        self = cls()
        self.index = index
        self._name_bytes = mesh.submesh_names.get(index, b'')
        self._material_bytes = sm.material
        self.submesh_name = _decode(self._name_bytes)
        self.material = _decode(sm.material)
        self.use_32bit_indexes = sm.use_32bit_indexes
        self.use_shared_vertices = sm.use_shared_vertices
        self._vd = mesh.shared_vertex_data if sm.use_shared_vertices else sm.vertex_data
        self._indices = sm.indices.astype(np.int64)
        self.faces = self._indices.tolist()
        self.face_count = len(self._indices) // 3
        assignments = sm.bone_assignments or (mesh.bone_assignments if sm.use_shared_vertices else [])
        self.boneassignments = [BoneAssignmentData(v, b, w) for v, b, w in assignments]
        target = 0 if sm.use_shared_vertices else index + 1
        self._poses = [p for p in mesh.poses if p.target == target]
        return self

    @property
    def encorded_name(self) -> bytes:
        return self._name_bytes if self._name_bytes is not None else self.submesh_name.encode('utf-8')

    @property
    def encorded_material(self) -> bytes:
        return self._material_bytes if self._material_bytes is not None else self.material.encode('utf-8')

    @property
    def geometry(self) -> GeometryData:
        return GeometryData(self._vd, self.use_shared_vertices)

    def _element(self, semantic, index=0):
        if self._vd is None:
            return None
        e = self._vd.find(semantic, index)
        return None if e is None else self._vd.read_element(e)

    def get_positions(self) -> np.ndarray:
        p = self._element(MF.VES_POSITION)
        if p is None:
            return np.zeros(0, dtype=np.float32)
        return to_blender(p[:, :3].astype(np.float64)).astype(np.float32).ravel()

    def get_normals(self) -> List[List[float]]:
        """Per face corner normals."""
        n = self._element(MF.VES_NORMAL)
        if n is None:
            return []
        return to_blender(n[self._indices, :3].astype(np.float64)).tolist()

    def get_texcoords(self) -> np.ndarray:
        """(uv set count, corner count * 2): per face corner UVs with V flipped."""
        sets = [e for e in self._vd.elements if e.semantic == MF.VES_TEXTURE_COORDINATES] if self._vd else []
        sets.sort(key=lambda e: e.index)
        out = np.zeros((len(sets), len(self._indices) * 2), dtype=np.float32)
        for i, e in enumerate(sets):
            uv = self._vd.read_element(e).astype(np.float32)
            if uv.shape[1] < 2:
                uv = np.c_[uv, np.zeros(len(uv), np.float32)]
            corner = uv[self._indices, :2]
            out[i, 0::2] = corner[:, 0]
            out[i, 1::2] = 1.0 - corner[:, 1]
        return out

    def get_colors(self, is_rgba: bool = True):
        """Per corner colours and alphas, flattened RGBA floats. The packed colour is read as ARGB."""
        e = self._vd.find(MF.VES_DIFFUSE) if self._vd else None
        if e is None:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty
        raw = self._vd.read_element(e)
        if raw.dtype != np.uint8:
            raw = raw.astype(np.uint8)
        raw = raw[self._indices].astype(np.float32) / 255.0
        b, g, r, a = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        colors = np.stack([r, g, b, a], axis=1) if is_rgba else np.stack([b, g, r, a], axis=1)
        alphas = np.stack([a, a, a, a], axis=1)
        return colors.ravel(), alphas.ravel()

    def get_shapekeys(self):
        base = self._element(MF.VES_POSITION)
        out = []
        if base is None:
            return out
        base = base[:, :3].astype(np.float64)
        for pose in self._poses:
            pos = base.copy()
            pos[pose.indices.astype(np.int64)] += pose.offsets
            out.append((_decode(pose.name), to_blender(pos).astype(np.float32)))
        return out

    def set_bone_mapping(self, mapping):
        if isinstance(mapping, dict):
            self._bone_map = {int(k): v for k, v in mapping.items()}
        else:
            self._bone_map = {b.id: b.name for b in mapping}

    def get_vertex_groups(self) -> List[VertexGroupData]:
        groups: Dict[int, list] = {}
        for ba in self.boneassignments:
            groups.setdefault(ba.bone_index, []).append(([ba.vertex_index], ba.weight))
        return [VertexGroupData(self._bone_map.get(bone, ''), groups[bone]) for bone in sorted(groups)]


class MeshData:
    def __init__(self, file: str = '', group: str = 'General'):
        self._file = file
        self._group = group
        self._submeshes: List[SubMeshData] = []
        self._skeleton_name = ''
        self._serializer = None
        self._bone_map: Dict[str, int] = {}
        self._source: Optional[MF.MeshFile] = None

    @classmethod
    def _from_file(cls, file: str, mesh: MF.MeshFile, serializer) -> 'MeshData':
        self = cls(file)
        self._source = mesh
        self._serializer = serializer
        self._skeleton_name = _decode(mesh.skeleton_name)
        self._submeshes = [SubMeshData._from_file(i, sm, mesh) for i, sm in enumerate(mesh.submeshes)]
        return self

    def get_submeshes(self) -> List[SubMeshData]:
        return list(self._submeshes)

    def set_submeshes(self, submeshes: List[SubMeshData]):
        self._submeshes = list(submeshes)

    def get_linked_skeleton_name(self) -> str:
        return self._skeleton_name

    def set_linked_skeleton_name(self, filename: str):
        self._skeleton_name = filename

    def get_linked_skeleton(self):
        if not self._skeleton_name or self._serializer is None:
            return None
        try:
            return self._serializer.load_skeleton(self._skeleton_name)
        except (OSError, ValueError) as exc:
            print('Failed to load linked skeleton:', exc)
            return None

    def set_bone_mapping(self, mapping):
        if isinstance(mapping, dict):
            self._bone_map = {v: int(k) for k, v in mapping.items()}
        else:
            self._bone_map = {b.name: b.id for b in mapping}

    def get_bone_id(self, bone_name: str) -> int:
        return self._bone_map.get(bone_name, 65535)

    def _to_file(self, version: str) -> MF.MeshFile:
        submeshes = [sm._to_file() for sm in self._submeshes]
        positions = [sm.vertex_data.read_element(sm.vertex_data.find(MF.VES_POSITION))
                     for sm in submeshes if sm.vertex_data is not None and sm.vertex_data.vertex_count]
        if positions:
            p = np.concatenate(positions).astype(np.float32)
            squared = p[:, 0] * p[:, 0] + p[:, 1] * p[:, 1] + p[:, 2] * p[:, 2]  # float32 like Ogre
            radius = float(np.sqrt(squared.max()))
            bounds = tuple(p.min(axis=0).tolist()) + tuple(p.max(axis=0).tolist()) + (radius,)
        else:
            bounds = (0.0,) * 7
        return MF.MeshFile(
            version=version,
            skeletally_animated=bool(self._skeleton_name),
            submeshes=submeshes,
            skeleton_name=self._skeleton_name.encode('utf-8') if self._skeleton_name else b'',
            bounds=bounds,
            submesh_names={i: sm.encorded_name for i, sm in enumerate(self._submeshes)},
            poses=[pose for sm in self._submeshes for pose in sm._poses])
