"""SkeletonData / AnimationData with the interface of the original native module.

Ogre bone space -> Blender bone space is the axis permutation P (Blender = (z, x, y)
of Ogre): keyframe rotations become P R P^T and translations are mapped by the
per bone matrix the add-on passes in.
"""
from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np

from . import _skeleton_file as SF
from ._math import Matrix3, OgreQuaternion, Vector3, matrices_to_quats

HELPER_BONE_ID = 65535
HELPER_BONE_LENGTH = 0.2


class SkeletonVersion(IntEnum):
    V_1_0 = 0
    V_1_8 = 1
    V_Latest = 100


class SkeletonAnimationBlendMode(IntEnum):
    average = 0
    cumulative = 1


def _decode(b: bytes) -> str:
    return b.decode('utf-8', errors='replace')


class BoneData:
    def __init__(self, id=0, name='', position=None, rotate=None, scale=None, parent_name='', child_names=None):
        self.id = int(id)
        self.name = name
        self.position = position if position is not None else Vector3(0, 0, 0)
        self.rotate = rotate if rotate is not None else OgreQuaternion(1, 0, 0, 0)
        self.scale = scale if scale is not None else Vector3(1, 1, 1)
        self.parent_name = parent_name
        self.child_names = list(child_names) if child_names is not None else []

    def __repr__(self):
        return f'BoneData({self.id}, {self.name!r}, parent={self.parent_name!r})'


class BlenderAnimationTrack:
    def __init__(self, name='', locations=None, rotations=None, scales=None, has_scale=False):
        self._name = name
        self._loc = locations if locations is not None else np.zeros((3, 0), np.float32)
        self._rot = rotations if rotations is not None else np.zeros((4, 0), np.float32)
        self._scl = scales if scales is not None else np.zeros((3, 0), np.float32)
        self._has_scale = has_scale

    @property
    def name(self) -> str:
        return self._name

    @property
    def has_scale(self) -> bool:
        return self._has_scale

    @property
    def nd_locations(self) -> np.ndarray:
        return self._loc

    @property
    def nd_rotations(self) -> np.ndarray:
        return self._rot

    @property
    def nd_scales(self) -> np.ndarray:
        return self._scl


def _interleave(frames: np.ndarray, values: np.ndarray) -> np.ndarray:
    """values (n, k) -> (k, 2n) of frame, value pairs for FCurve keyframe 'co'."""
    n, k = values.shape
    out = np.empty((k, 2 * n), dtype=np.float32)
    out[:, 0::2] = frames
    out[:, 1::2] = values.T
    return out


def _continuous(q: np.ndarray) -> np.ndarray:
    """Flip quaternions so consecutive keys stay in the same hemisphere."""
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] = -q[i]
    return q


class AnimationData:
    def __init__(self):
        self.name = ''
        self.length = 0.0
        self._tracks: List[SF.TrackFile] = []
        self._bone_names: Dict[int, str] = {}
        # export side: bone name -> TrackFile in Ogre space
        self._export_tracks: Dict[str, SF.TrackFile] = {}

    @classmethod
    def _from_file(cls, anim: SF.AnimationFile, bone_names: Dict[int, str]) -> 'AnimationData':
        self = cls()
        self.name = _decode(anim.name)
        self.length = anim.length
        self._tracks = anim.tracks
        self._bone_names = bone_names
        return self

    # ---------------------------------------------------------- import

    def get_animations(self, bone_matrix_map: Dict[str, Matrix3], fps: float, round_frame: bool) -> List[BlenderAnimationTrack]:
        out = []
        for track in self._tracks:
            name = self._bone_names.get(track.bone)
            if name is None or name not in bone_matrix_map:
                continue
            m = bone_matrix_map[name].m
            frames = (track.times * fps).astype(np.float32)
            if round_frame:
                frames = np.floor(frames + 0.5).astype(np.float32)
            loc = track.translations @ m.T
            q = _continuous(track.rotations)
            rot = q[:, [0, 3, 1, 2]]
            scl = track.scales[:, [2, 0, 1]]
            has_scale = bool(np.any(np.abs(track.scales - 1.0) > 1e-6))
            out.append(BlenderAnimationTrack(name, _interleave(frames, loc), _interleave(frames, rot),
                                             _interleave(frames, scl), has_scale))
        return out

    # ---------------------------------------------------------- export

    def append_animation_track(self, bone_name: str, bone_matrix: Matrix3, nd_times, nd_locations,
                               nd_rotations, nd_scales, use_scale: bool = False):
        """Blender pose bone channels (rows x/y/z, w/x/y/z, x/y/z) -> Ogre keyframes."""
        times = np.asarray(nd_times, dtype=np.float32).astype(np.float64)
        loc = np.asarray(nd_locations, dtype=np.float64).reshape(3, -1).T
        rot = np.asarray(nd_rotations, dtype=np.float64).reshape(4, -1).T
        scl = np.asarray(nd_scales, dtype=np.float64).reshape(3, -1).T
        self._add_track(bone_name, times, _transform32(bone_matrix.m, loc), rot[:, [0, 2, 3, 1]], scl, use_scale)

    def set_animation_tracks(self, bone_matrix_map, pose_matrix_map, time_array, use_scale: bool = False):
        """Baked local pose matrices (one 4x4 per time) -> Ogre keyframes."""
        times = np.asarray(time_array, dtype=np.float32).astype(np.float64)
        for name, matrices in pose_matrix_map.items():
            if name not in bone_matrix_map:
                continue
            mats = np.asarray(matrices, dtype=np.float64).reshape(-1, 4, 4)
            basis = mats[:, :3, :3]
            scl = np.linalg.norm(basis, axis=1)
            scl[scl == 0] = 1.0
            rot = matrices_to_quats(basis / scl[:, None, :])
            self._add_track(name, times, _transform32(bone_matrix_map[name].m, mats[:, :3, 3]),
                            rot[:, [0, 2, 3, 1]], scl, use_scale)

    def _add_track(self, name, times, translations, rotations, scales, use_scale):
        # Blender bone axes (x, y, z) are Ogre bone axes (z, x, y), as for the rotations.
        # The original module permuted scales inconsistently between its import and its two
        # export paths; here every path uses the rotation's axis mapping.
        scales = scales[:, [1, 2, 0]] if use_scale else np.ones_like(translations)
        self._export_tracks[name] = SF.TrackFile(-1, times, rotations, translations, scales)


def _transform32(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rows of v multiplied by m in float32, the way Ogre::Matrix3 * Vector3 rounds."""
    m = m.astype(np.float32)
    v = v.astype(np.float32)
    out = np.empty_like(v)
    for row in range(3):
        out[:, row] = m[row, 0] * v[:, 0] + m[row, 1] * v[:, 1] + m[row, 2] * v[:, 2]
    return out.astype(np.float64)


class SkeletonData:
    def __init__(self, file: str = '', group: str = 'General'):
        self._file = file
        self._group = group
        self._source: Optional[SF.SkeletonFile] = None
        self._bones: List[BoneData] = []
        self._animations: List[AnimationData] = []
        self.blend_mode = SkeletonAnimationBlendMode.average

    @classmethod
    def _from_file(cls, file: str, skel: SF.SkeletonFile) -> 'SkeletonData':
        self = cls(file)
        self._source = skel
        self.blend_mode = SkeletonAnimationBlendMode(skel.blend_mode) if skel.blend_mode in (0, 1) \
            else SkeletonAnimationBlendMode.average
        by_handle = {b.handle: b for b in skel.bones}
        bones = []
        for b in skel.bones:
            w, x, y, z = b.orientation
            parent = by_handle.get(b.parent)
            bones.append(BoneData(b.handle, _decode(b.name), Vector3(*b.position), OgreQuaternion(w, x, y, z),
                                  Vector3(*b.scale), _decode(parent.name) if parent else '',
                                  [_decode(c.name) for c in skel.bones if c.parent == b.handle]))
        self._bones = bones
        names = {b.handle: _decode(b.name) for b in skel.bones}
        self._animations = [AnimationData._from_file(a, names) for a in skel.animations]
        return self

    def get_bones(self, has_helper: bool = True) -> List[BoneData]:
        bones = [BoneData(b.id, b.name, Vector3(b.position.x, b.position.y, b.position.z),
                          OgreQuaternion(b.rotate.w, b.rotate.x, b.rotate.y, b.rotate.z),
                          Vector3(b.scale.x, b.scale.y, b.scale.z), b.parent_name, b.child_names)
                 for b in self._bones]
        if not has_helper:
            return bones
        helpers = []
        # Helper bones give leaf and branching bones a tail direction on import
        for b in bones:
            if len(b.child_names) != 1:
                helpers.append(self._helper('Helper' + b.name, b.name))
        # Bones sitting exactly on their parent get one more helper, listed under the parent
        by_name = {b.name: b for b in bones}
        for b in bones:
            if b.parent_name and (b.position.x, b.position.y, b.position.z) == (0.0, 0.0, 0.0):
                helpers.append(self._helper('HelperZero' + b.name, b.name))
                by_name[b.parent_name].child_names.append('HelperZero' + b.name)
        return bones + helpers

    @staticmethod
    def _helper(name: str, parent: str) -> BoneData:
        return BoneData(HELPER_BONE_ID, name, Vector3(HELPER_BONE_LENGTH, 0, 0), OgreQuaternion(1, 0, 0, 0),
                        Vector3(1, 1, 1), parent, [])

    def set_bones(self, bones: List[BoneData]):
        self._bones = list(bones)

    def get_animations(self) -> List[AnimationData]:
        return list(self._animations)

    def add_animation(self, animation: AnimationData):
        self._animations.append(animation)

    def calc_animation_fps(self) -> float:
        """Keyframe rate of the first animation, from its shortest keyframe interval."""
        for anim in self._animations:
            if not anim._tracks:
                continue
            dts = [np.diff(t.times) for t in anim._tracks if len(t.times) > 1]
            dts = np.concatenate(dts) if dts else np.zeros(0)
            dts = dts[dts > 0]
            return float(1.0 / dts.min()) if len(dts) else 0.0
        return 0.0

    def _to_file(self, version: str) -> SF.SkeletonFile:
        bones = sorted(self._bones, key=lambda b: b.id)
        handles = {b.name: b.id for b in bones}
        out = SF.SkeletonFile(version=version, blend_mode=int(self.blend_mode))
        for b in bones:
            out.bones.append(SF.BoneFile(b.name.encode('utf-8'), b.id,
                                         (b.position.x, b.position.y, b.position.z),
                                         (b.rotate.w, b.rotate.x, b.rotate.y, b.rotate.z),
                                         (b.scale.x, b.scale.y, b.scale.z),
                                         handles.get(b.parent_name, -1)))
        for anim in self._animations:
            tracks = anim._tracks if not anim._export_tracks else []
            for name, track in anim._export_tracks.items():
                if name in handles and _has_non_identity_keys(track):
                    keep = _keyframes_to_keep(track)
                    tracks.append(SF.TrackFile(handles[name], track.times[keep], track.rotations[keep],
                                               track.translations[keep], track.scales[keep]))
            out.animations.append(SF.AnimationFile(anim.name.encode('utf-8'), anim.length,
                                                   sorted(tracks, key=lambda t: t.bone)))
        return out


def _keyframes_to_keep(track: SF.TrackFile, tolerance: float = 1e-3) -> np.ndarray:
    """Ogre::NodeAnimationTrack::optimise: in a run of 5+ equal keyframes only the two keys at
    each end are kept."""
    remove = []
    last = None
    duplicates = 0
    for k in range(len(track.times)):
        key = (track.translations[k], track.scales[k], track.rotations[k])
        if last is not None and _key_equals(key, last, tolerance):
            duplicates += 1
            if duplicates == 4:
                remove.append(k - 2)
                duplicates -= 1
        else:
            duplicates = 0
            last = key
    keep = np.ones(len(track.times), dtype=bool)
    keep[remove] = False
    return keep


def _key_equals(a, b, tolerance) -> bool:
    if np.any(np.abs(a[0] - b[0]) > tolerance) or np.any(np.abs(a[1] - b[1]) > tolerance):
        return False
    d = float(np.dot(a[2], b[2]))
    return abs(np.arccos(np.clip(2.0 * d * d - 1.0, -1.0, 1.0))) <= tolerance


def _has_non_identity_keys(track: SF.TrackFile, tolerance: float = 1e-3) -> bool:
    """Same test as Ogre::NodeAnimationTrack::hasNonZeroKeyFrames, used to drop empty tracks."""
    if np.any(np.abs(track.translations) > tolerance) or np.any(np.abs(track.scales - 1.0) > tolerance):
        return True
    w = np.clip(track.rotations[:, 0], -1.0, 1.0)
    return bool(np.any(2.0 * np.arccos(w) > tolerance))
