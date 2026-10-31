import pickle, glob, sys, zlib, numpy as np
sys.path.insert(0, 'addon')
from kenshi_blender_tool import _skeleton_file as SF
from kenshi_blender_tool._skeleton import SkeletonData
from kenshi_blender_tool._math import Matrix3
def rot(name):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]
fails = {}
def chk(k, ok, f, info=''):
    if not ok: fails.setdefault(k, []).append((f[38:], info))
pk = sorted(glob.glob(sys.argv[1] + '/*.pkl'))
for p in pk:
    d = pickle.load(open(p, 'rb')); f = d['file']
    sk = SkeletonData._from_file(f, SF.read_skeleton(open(f, 'rb').read()))
    bl = lambda bs: [(b.id, b.name, b.parent_name, list(b.child_names), (b.position.x, b.position.y, b.position.z), (b.rotate.w, b.rotate.x, b.rotate.y, b.rotate.z), (b.scale.x, b.scale.y, b.scale.z)) for b in bs]
    for key, mine in (('bones', bl(sk.get_bones(True))), ('bones_nohelp', bl(sk.get_bones(False)))):
        ok = len(mine) == len(d[key]) and all(a[:3] == b[:3] and sorted(a[3]) == sorted(b[3]) and np.allclose(a[4], b[4], atol=1e-5) and np.allclose(a[5], b[5], atol=1e-5) and np.allclose(a[6], b[6], atol=1e-5) for a, b in zip(mine, d[key]))
        chk(key, ok, f)
    chk('fps', abs(sk.calc_animation_fps() - d['fps']) < 2e-3, f, (sk.calc_animation_fps(), d['fps']))
    anims = sk.get_animations()
    chk('nanim', [(a.name, round(a.length, 5)) for a in anims] == [(a['name'], round(a['length'], 5)) for a in d['anims']], f)
    mm = {b.name: Matrix3(rot(b.name)) for b in sk.get_bones(False)}
    for a, e in zip(anims, d['anims']):
        if 'tracks' not in e: continue
        mt = a.get_animations(mm, 24.0, False)
        chk('track_names', [t.name for t in mt] == [t[0] for t in e['tracks']], f, (a.name, len(mt), len(e['tracks'])))
        for t, o in zip(mt, e['tracks']):
            if t.name != o[0]: continue
            chk('has_scale', t.has_scale == o[1], f)
            chk('loc', t.nd_locations.shape == o[2].shape and np.allclose(t.nd_locations, o[2], atol=1e-4), f, (a.name, t.name))
            ok = t.nd_rotations.shape == o[3].shape and np.allclose(t.nd_rotations, o[3], atol=1e-4)
            chk('rot', ok, f, (a.name, t.name))
            if o[1]: chk('scl', np.allclose(t.nd_scales, o[4], atol=1e-4), f)
        mr = a.get_animations(mm, 30.0, True)
        for t, o in zip(mr, e['tracks_round']):
            chk('round', t.nd_locations.shape == o[1].shape and np.allclose(t.nd_locations, o[1], atol=1e-4) and np.allclose(t.nd_rotations, o[2], atol=1e-4), f, (a.name, t.name, t.nd_locations.shape, o[1].shape))
print(len(pk), 'files')
for k, v in fails.items(): print('FAIL', k, len(v), v[:3])
