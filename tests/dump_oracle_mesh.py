"""Run with a numpy 1.x Python 3.10: dumps what the original module returns for meshes."""
import sys, os, pickle, glob, random, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from oracle import K
from config import KENSHI_DATA, LOG  # noqa: E402
out_dir = sys.argv[1]; files = [l for l in open(sys.argv[2]).read().splitlines() if l]
os.makedirs(out_dir, exist_ok=True)
BONEMAP = {i: f'b{i}' for i in range(300)}
for n, f in enumerate(files):
    try:
        ser = K.KenshiObjectSerializer(LOG)
        ser.add_resource_location(os.path.dirname(f))
        m = ser.load_mesh(os.path.basename(f))
    except Exception as e:
        print('ERR', f, e); continue
    d = {'file': f, 'skel': m.get_linked_skeleton_name(), 'subs': []}
    for sm in m.get_submeshes():
        g = sm.geometry
        s = dict(index=sm.index, name=sm.encorded_name, material=sm.encorded_material, face_count=sm.face_count,
                 vertex_count=g.vertex_count, flags=(g.has_normals, g.has_colors, g.has_texture_coord, g.texcoords_size,
                 g.has_tangents, g.tangent_dimensions, g.has_binormals, g.has_positions, g.has_shared_geometry),
                 idx32=sm.use_32bit_indexes, shared=sm.use_shared_vertices,
                 faces=np.array(sm.faces), positions=np.array(sm.get_positions()),
                 normals=np.array(sm.get_normals()) if g.has_normals else None,
                 texcoords=np.array(sm.get_texcoords()),
                 shapekeys=[(nm, np.array(x)) for nm, x in sm.get_shapekeys()],
                 ba=[(b.vertex_index, b.bone_index, b.weight) for b in sm.boneassignments])
        if g.has_colors:
            c, a = sm.get_colors(); s['colors'] = np.array(c); s['alphas'] = np.array(a)
        sm.set_bone_mapping(BONEMAP)
        s['vg'] = [(v.name, v.group) for v in sm.get_vertex_groups()]
        d['subs'].append(s)
    del m, ser
    pickle.dump(d, open(os.path.join(out_dir, f'{n:05d}.pkl'), 'wb'))
print('done', len(files))
