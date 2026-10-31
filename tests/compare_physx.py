import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'addon'))
from kenshi_blender_tool._physx import KenshiPhysXSerializer
res = pickle.load(open(sys.argv[1], 'rb'))
px = KenshiPhysXSerializer()
for kind, pts, tris, ref in res['export']:
    mine = px.export_convex_hull(pts) if kind == 'convex' else px.export_triangle_mesh(pts, tris)
    print(kind, len(mine), len(ref), 'equal' if mine == ref else ('prefix-equal' if mine.startswith(ref) else 'DIFF'))
bad = 0
for f, kind, hexs, pts, tris in res['import']:
    c = px.import_convex_hull(hexs) if kind == 'Convex' else px.import_triangle_mesh(hexs)
    ok = len(c.points) == len(pts) and np.allclose(np.array(c.points).reshape(-1, 3), np.array(pts).reshape(-1, 3)) and (c.triangles == tris)
    if not ok:
        bad += 1
        if bad < 4: print('import diff', f[38:], kind, len(c.points), len(pts), len(c.triangles), len(tris), c.triangles[:6], tris[:6])
print('imports', len(res['import']), 'bad', bad)
