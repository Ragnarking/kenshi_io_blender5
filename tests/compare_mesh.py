import pickle, glob, sys, numpy as np
sys.path.insert(0, 'addon')
from kenshi_blender_tool import _mesh_file as MF
from kenshi_blender_tool._mesh import MeshData
BONEMAP = {i: f'b{i}' for i in range(300)}
fails = {}
def chk(k, ok, f):
    if not ok: fails.setdefault(k, []).append(f)
pk = sorted(glob.glob(sys.argv[1] + '/*.pkl'))
for p in pk:
    d = pickle.load(open(p, 'rb')); f = d['file']
    m = MeshData._from_file(f, MF.read_mesh(open(f, 'rb').read()), None)
    chk('skel', m.get_linked_skeleton_name() == d['skel'], f)
    subs = m.get_submeshes(); chk('nsub', len(subs) == len(d['subs']), f)
    for sm, s in zip(subs, d['subs']):
        g = sm.geometry
        chk('names', (sm.index, sm.encorded_name, sm.encorded_material, sm.face_count, g.vertex_count) == (s['index'], s['name'], s['material'], s['face_count'], s['vertex_count']), f)
        flags = (g.has_normals, g.has_colors, g.has_texture_coord, g.texcoords_size, g.has_tangents, g.tangent_dimensions, g.has_binormals, g.has_positions, g.has_shared_geometry)
        chk('flags', flags == s['flags'], f)
        chk('faces', np.array_equal(np.array(sm.faces), s['faces']), f)
        chk('pos', np.allclose(sm.get_positions(), s['positions'], atol=1e-5), f)
        if s['normals'] is not None: chk('nrm', np.allclose(np.array(sm.get_normals()), s['normals'], atol=1e-5), f)
        t = sm.get_texcoords(); chk('uv', t.shape == s['texcoords'].shape and np.allclose(t, s['texcoords'], atol=1e-6), f)
        if 'colors' in s:
            c, a = sm.get_colors(); chk('col', np.allclose(c, s['colors'], atol=1e-6) and np.allclose(a, s['alphas'], atol=1e-6), f)
        sk = sm.get_shapekeys(); chk('sk', len(sk) == len(s['shapekeys']) and all(n1 == n2 and np.allclose(x1, x2, atol=1e-5) for (n1, x1), (n2, x2) in zip(sk, s['shapekeys'])), f)
        chk('ba', [(b.vertex_index, b.bone_index) for b in sm.boneassignments] == [b[:2] for b in s['ba']], f)
        sm.set_bone_mapping(BONEMAP)
        vg = [(v.name, v.group) for v in sm.get_vertex_groups()]
        chk('vg', [n for n, _ in vg] == [n for n, _ in s['vg']] and all([x[0] for x in g1] == [x[0] for x in g2] and np.allclose([x[1] for x in g1], [x[1] for x in g2]) for (_, g1), (_, g2) in zip(vg, s['vg'])), f)
print(len(pk), 'files')
for k, v in fails.items(): print('FAIL', k, len(v), v[:3])
