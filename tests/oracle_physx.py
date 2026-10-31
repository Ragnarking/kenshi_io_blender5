import sys, os, pickle, glob, re, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from oracle import K
from config import KENSHI_DATA, LOG  # noqa: E402
out = sys.argv[1]
px = K.KenshiPhysXSerializer()
rng = np.random.default_rng(5)
res = {'export': [], 'import': []}
cube = [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
pts = (rng.normal(size=(40, 3))).tolist()
tris = [0, 1, 2, 2, 1, 3, 4, 5, 6, 6, 5, 7, 0, 4, 1, 1, 4, 5]
for p in (cube, pts):
    res['export'].append(('convex', p, None, px.export_convex_hull(p)))
res['export'].append(('mesh', cube, tris, px.export_triangle_mesh(cube, tris)))
grid = [[x, y, (x * y) % 3 * 0.1] for x in range(20) for y in range(20)]
gt = [i for x in range(19) for y in range(19) for i in (x*20+y, x*20+y+1, (x+1)*20+y, (x+1)*20+y, x*20+y+1, (x+1)*20+y+1)]
res['export'].append(('mesh', grid, gt, px.export_triangle_mesh(grid, gt)))
files = [f for f in sorted(glob.glob(KENSHI_DATA + '/**/*.xml', recursive=True))
         if 'cookedData' in open(f, encoding='latin1').read()][::7]
n = 0
for f in files:
    t = open(f, encoding='latin1').read()
    found = re.findall(r'<Nx(Convex|Triangle)MeshDesc[^>]*>.*?<cookedData>([0-9A-Fa-f\s]+)</cookedData>', t, re.S)
    for kind, hexs in ((k, re.sub(r'\s', '', h)) for k, h in found):
        c = px.import_convex_hull(hexs) if kind == 'Convex' else px.import_triangle_mesh(hexs)
        res['import'].append((f, kind, hexs, [list(q) for q in c.points], list(c.triangles)))
        n += 1
    if n > 150: break
pickle.dump(res, open(out, 'wb'))
print('exports', [len(e[3]) for e in res['export']], 'imports', n, res['export'][0][3][:60])
