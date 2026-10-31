"""Synthetic skeleton export inputs, written to work with both the oracle and the new module."""
import numpy as np


def rot_matrix(rng):
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def build(K, seed, use_scale, bake, n_frames=12):
    """K is the module (oracle or new). Returns a SkeletonData ready to save."""
    rng = np.random.default_rng(seed)
    names = [f'bone{i}' for i in range(6)]
    parents = ['', 'bone0', 'bone1', 'bone1', 'bone3', 'bone0']
    bones = []
    for i, n in enumerate(names):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        p = rng.normal(size=3)
        bones.append(K.BoneData(i, n, K.Vector3(*p), K.OgreQuaternion(*q), K.Vector3(1, 1, 1), parents[i], []))
    return bones, names, rng


def make_animation(K, rng, names, use_scale, bake, name, n_frames=12, fps=24.0):
    a = K.AnimationData()
    a.name = name
    a.length = (n_frames - 1) / fps
    times = (np.arange(n_frames, dtype=np.float32)) / fps
    fix = {n: rot_matrix(rng) for n in names}
    if not bake:
        for n in names:
            loc = rng.normal(size=(3, n_frames)) * 0.3
            q = rng.normal(size=(4, n_frames)); q /= np.linalg.norm(q, axis=0)
            scl = 1.0 + (rng.normal(size=(3, n_frames)) * 0.1 if use_scale else 0)
            scl = np.broadcast_to(scl, (3, n_frames)).astype(np.float64)
            if n == 'bone5':  # constant track
                loc[:] = 0; q[:] = np.array([[1], [0], [0], [0]]); scl = np.ones((3, n_frames))
            a.append_animation_track(bone_name=n, bone_matrix=K.Matrix3(fix[n].tolist()),
                                     nd_times=times, nd_locations=loc.astype(np.float32).astype(object),
                                     nd_rotations=q.astype(np.float32).astype(object),
                                     nd_scales=scl.astype(np.float32).astype(object), use_scale=use_scale)
    else:
        pose = {}
        for n in names:
            mats = []
            for f in range(n_frames):
                m = np.eye(4)
                m[:3, :3] = rot_matrix(rng) * (1.0 + (rng.normal(size=3) * 0.1 if use_scale else 0))
                m[:3, 3] = rng.normal(size=3) * 0.3
                mats.append(m.tolist())
            pose[n] = mats
        a.set_animation_tracks(bone_matrix_map={n: K.Matrix3(fix[n].tolist()) for n in names},
                               pose_matrix_map=pose, time_array=times.tolist(), use_scale=use_scale)
    return a


CASES = [dict(seed=1, use_scale=False, bake=False), dict(seed=2, use_scale=True, bake=False),
         dict(seed=3, use_scale=False, bake=True), dict(seed=4, use_scale=True, bake=True)]


def build_skeleton(K, ser, case):
    bones, names, rng = build(K, case['seed'], case['use_scale'], case['bake'])
    sk = ser.create_skeleton('case.skeleton')
    sk.set_bones(bones)
    for i in range(2):
        sk.add_animation(make_animation(K, rng, names, case['use_scale'], case['bake'], f'anim{i}'))
    return sk
