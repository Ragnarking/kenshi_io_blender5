import sys, os, pickle, glob, numpy as np, zlib
sys.path.insert(0, os.path.dirname(__file__))
from oracle import K
from config import KENSHI_DATA, LOG  # noqa: E402
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
files = sorted(glob.glob(KENSHI_DATA + '/**/*.skeleton', recursive=True))
def rot(name):
    rng = np.random.default_rng(zlib.crc32(name.encode()))
    q = rng.normal(size=4); q /= np.linalg.norm(q); w, x, y, z = q
    return [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
for n, f in enumerate(files):
    if n < start: continue
    try:
        s = K.KenshiObjectSerializer(LOG); s.add_resource_location(os.path.dirname(f))
        sk = s.load_skeleton(f)
    except Exception as e:
        print('ERR', f, e); continue
    bl = lambda bs: [(b.id, b.name, b.parent_name, list(b.child_names), (b.position.x, b.position.y, b.position.z), (b.rotate.w, b.rotate.x, b.rotate.y, b.rotate.z), (b.scale.x, b.scale.y, b.scale.z)) for b in bs]
    d = {'file': f, 'bones': bl(sk.get_bones(True)), 'bones_nohelp': bl(sk.get_bones(False)), 'fps': sk.calc_animation_fps(), 'anims': []}
    mm = {b.name: K.Matrix3(rot(b.name)) for b in sk.get_bones(False)}
    for i, a in enumerate(sk.get_animations()):
        e = {'name': a.name, 'length': a.length}
        if n % 6 == 0 and i < 4:
            e['tracks'] = [(t.name, t.has_scale, np.array(t.nd_locations), np.array(t.nd_rotations), np.array(t.nd_scales)) for t in a.get_animations(mm, 24.0, False)]
            e['tracks_round'] = [(t.name, np.array(t.nd_locations), np.array(t.nd_rotations)) for t in a.get_animations(mm, 30.0, True)]
        d['anims'].append(e)
    del sk, s
    pickle.dump(d, open(os.path.join(out, f'{n:05d}.pkl'), 'wb'))
print('done')
