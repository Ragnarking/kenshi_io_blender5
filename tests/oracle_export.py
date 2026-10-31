import sys, os, pickle, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from oracle import K
from export_cases import CASES, OPTIMIZE, make_case
from config import KENSHI_DATA, LOG  # noqa: E402
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
res = {}
for ci, c in enumerate(CASES):
    for opt in OPTIMIZE:
        for vn in ('V_Latest', 'V_1_8', 'V_1_4'):
            ser = K.KenshiObjectSerializer(LOG)
            m = ser.create_mesh(f'case{ci}.mesh')
            bones = [K.BoneData(i, f'bone{i}', K.Vector3(i, 0, 0), K.OgreQuaternion(1, 0, 0, 0), K.Vector3(1, 1, 1), 'bone0' if i else '', []) for i in range(5)]
            m.set_bone_mapping(bones)
            subs = []
            for si in range(2):
                d, ba, sk = make_case(c['seed'] * 10 + si, **{k: v for k, v in c.items() if k != 'seed'})
                s = K.SubMeshData(); s.index = si; s.submesh_name = f'sub{si}'; s.material = f'mat{si}'
                idx = s.set_vertex(optimize=opt, **d)
                res[(ci, opt, si)] = np.array(idx)
                for name, off in sk: s.append_shapekey(name, off, idx)
                s.set_bone_assignments([K.BoneAssignmentData(*b) for b in ba], idx)
                subs.append(s)
            m.set_submeshes(subs)
            if ci != 4:
                m.set_linked_skeleton_name('case.skeleton')
            ser.save_mesh(m, os.path.join(out, f'case{ci}_{int(opt)}_{vn}.mesh'), getattr(K.MeshVersion, vn))
            del m, ser
pickle.dump(res, open(os.path.join(out, 'indices.pkl'), 'wb'))
print('ok')
