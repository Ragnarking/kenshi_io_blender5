"""Binary Ogre .skeleton reader/writer (formats v1.10 and v1.80)."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from ._stream import Reader, Writer, CHUNK_HEADER_SIZE

SKELETON_HEADER = 0x1000
SKELETON_BLENDMODE = 0x1010
SKELETON_BONE = 0x2000
SKELETON_BONE_PARENT = 0x3000
SKELETON_ANIMATION = 0x4000
SKELETON_ANIMATION_BASEINFO = 0x4010
SKELETON_ANIMATION_TRACK = 0x4100
SKELETON_ANIMATION_TRACK_KEYFRAME = 0x4110
SKELETON_ANIMATION_LINK = 0x5000

VERSION_1_0 = '[Serializer_v1.10]'
VERSION_1_8 = '[Serializer_v1.80]'

# bone chunk without scale: name + '\n' + handle + position + orientation
_KEYFRAME_SIZE_NO_SCALE = CHUNK_HEADER_SIZE + 4 + 16 + 12
# bone chunk without scale, as computed by Ogre: header + handle + position + orientation
_BONE_SIZE_NO_SCALE = CHUNK_HEADER_SIZE + 2 + 12 + 16


@dataclass
class BoneFile:
    name: bytes
    handle: int
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # w, x, y, z
    scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    parent: int = -1


@dataclass
class TrackFile:
    bone: int
    times: np.ndarray            # (n,)
    rotations: np.ndarray        # (n, 4) w, x, y, z
    translations: np.ndarray     # (n, 3)
    scales: np.ndarray           # (n, 3)


@dataclass
class AnimationFile:
    name: bytes
    length: float
    tracks: List[TrackFile] = field(default_factory=list)
    base_name: Optional[bytes] = None
    base_time: float = 0.0


@dataclass
class SkeletonFile:
    version: str = VERSION_1_8
    blend_mode: int = 0
    bones: List[BoneFile] = field(default_factory=list)
    animations: List[AnimationFile] = field(default_factory=list)
    links: List[Tuple[bytes, float]] = field(default_factory=list)


def _quat(r: Reader):
    x, y, z, w = r.floats(4)
    return (w, x, y, z)


def read_skeleton(data: bytes) -> SkeletonFile:
    """Chunk lengths are only used to detect optional scales, like Ogre does.
    Several exporters wrote lengths without the 6 byte chunk header."""
    r = Reader(data)
    if r.u16() != SKELETON_HEADER:
        raise ValueError('File header not found, not an Ogre skeleton')
    version = r.string().decode('ascii', errors='replace')
    if version not in (VERSION_1_0, VERSION_1_8):
        raise ValueError(f'Unsupported skeleton version {version}')
    skel = SkeletonFile(version=version)
    bones_by_handle = {}
    while r.pos + CHUNK_HEADER_SIZE <= len(r.data):
        cid, start, end = r.chunk()
        length = end - start
        if cid == SKELETON_BLENDMODE:
            skel.blend_mode = r.u16()
        elif cid == SKELETON_BONE:
            name = r.string()
            handle = r.u16()
            pos = r.floats(3)
            rot = _quat(r)
            # Ogre leaves the name out of this size, which keeps old files loading
            scale = r.floats(3) if length > _BONE_SIZE_NO_SCALE else (1.0, 1.0, 1.0)
            bone = BoneFile(name, handle, pos, rot, scale)
            skel.bones.append(bone)
            bones_by_handle[handle] = bone
        elif cid == SKELETON_BONE_PARENT:
            child, parent = r.unpack('2H')
            bones_by_handle[child].parent = parent
        elif cid == SKELETON_ANIMATION:
            skel.animations.append(_read_animation(r))
        elif cid == SKELETON_ANIMATION_LINK:
            skel.links.append((r.string(), r.f32()))
        else:
            raise ValueError(f'Unknown skeleton chunk {cid:#x} at {start}')
    return skel


def _read_animation(r: Reader) -> AnimationFile:
    anim = AnimationFile(name=r.string(), length=r.f32())
    while True:
        cid = r.peek_chunk_id()
        if cid == SKELETON_ANIMATION_BASEINFO:
            r.chunk()
            anim.base_name = r.string()
            anim.base_time = r.f32()
        elif cid == SKELETON_ANIMATION_TRACK:
            r.chunk()
            anim.tracks.append(_read_track(r))
        else:
            return anim


def _read_track(r: Reader) -> TrackFile:
    bone = r.u16()
    times, rots, trans, scales = [], [], [], []
    while r.peek_chunk_id() == SKELETON_ANIMATION_TRACK_KEYFRAME:
        _, start, end = r.chunk()
        times.append(r.f32())
        rots.append(_quat(r))
        trans.append(r.floats(3))
        scales.append(r.floats(3) if end - start > _KEYFRAME_SIZE_NO_SCALE else (1.0, 1.0, 1.0))
    return TrackFile(bone,
                     np.array(times, dtype=np.float64),
                     np.array(rots, dtype=np.float64).reshape(-1, 4),
                     np.array(trans, dtype=np.float64).reshape(-1, 3),
                     np.array(scales, dtype=np.float64).reshape(-1, 3))


# ---------------------------------------------------------------- writing

def write_skeleton(skel: SkeletonFile) -> bytes:
    w = Writer()
    w.u16(SKELETON_HEADER)
    w.string(skel.version)
    if skel.version != VERSION_1_0:
        w.begin_chunk(SKELETON_BLENDMODE)
        w.u16(skel.blend_mode)
        w.end_chunk()
    for bone in skel.bones:
        w.begin_chunk(SKELETON_BONE)
        w.string(bone.name)
        w.u16(bone.handle)
        w.floats(*bone.position)
        bw, bx, by, bz = bone.orientation
        w.floats(bx, by, bz, bw)
        has_scale = tuple(bone.scale) != (1.0, 1.0, 1.0)
        if has_scale:
            w.floats(*bone.scale)
        # Ogre's SkeletonSerializer leaves the name out of the bone chunk size
        w.end_chunk(_BONE_SIZE_NO_SCALE + (12 if has_scale else 0))
    for bone in skel.bones:
        if bone.parent >= 0:
            w.begin_chunk(SKELETON_BONE_PARENT)
            w.pack('2H', bone.handle, bone.parent)
            w.end_chunk()
    for anim in skel.animations:
        w.begin_chunk(SKELETON_ANIMATION)
        w.string(anim.name)
        w.f32(anim.length)
        if anim.base_name is not None:
            w.begin_chunk(SKELETON_ANIMATION_BASEINFO)
            w.string(anim.base_name)
            w.f32(anim.base_time)
            w.end_chunk()
        for track in anim.tracks:
            w.begin_chunk(SKELETON_ANIMATION_TRACK)
            w.u16(track.bone)
            for i in range(len(track.times)):
                w.begin_chunk(SKELETON_ANIMATION_TRACK_KEYFRAME)
                w.f32(track.times[i])
                qw, qx, qy, qz = track.rotations[i]
                w.floats(qx, qy, qz, qw)
                w.floats(*track.translations[i])
                s = tuple(float(np.float32(v)) for v in track.scales[i])
                if s != (1.0, 1.0, 1.0):
                    w.floats(*s)
                w.end_chunk()
            w.end_chunk()
        w.end_chunk()
    for name, scale in skel.links:
        w.begin_chunk(SKELETON_ANIMATION_LINK)
        w.string(name)
        w.f32(scale)
        w.end_chunk()
    return w.getvalue()
