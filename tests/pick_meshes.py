import glob, random, sys
sys.path.insert(0, 'addon')
from kenshi_blender_tool import _mesh_file as MF
from config import KENSHI_DATA, LOG  # noqa: E402
files = sorted(glob.glob(KENSHI_DATA + '/**/*.mesh', recursive=True))
random.seed(1); pick = set(random.sample(files, 350)); seen = set()
for f in files:
    m = MF.read_mesh(open(f, 'rb').read())
    keys = [m.version, bool(m.poses), bool(m.skeleton_name)]
    for sm in m.submeshes:
        keys.append(tuple((e.source, e.type, e.semantic, e.index) for e in sm.vertex_data.elements)); keys.append(sm.use_32bit_indexes)
    for k in keys:
        if (k, ) not in seen: seen.add((k,)); pick.add(f)
print('\n'.join(sorted(pick)))
