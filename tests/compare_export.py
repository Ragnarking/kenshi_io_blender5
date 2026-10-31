"""Builds the synthetic export cases with the new module and compares the files with the oracle ones."""
import os, sys, pickle, contextlib, io
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'addon'))
sys.path.insert(0, os.path.dirname(__file__))
from kenshi_blender_tool import _mesh_file as MF
from kenshi_blender_tool._mesh import MeshData, SubMeshData, BoneAssignmentData
from kenshi_blender_tool._skeleton import BoneData
from kenshi_blender_tool._math import Vector3, OgreQuaternion
from export_cases import CASES, OPTIMIZE, make_case

ref = sys.argv[1]
VERSIONS = {'V_Latest': MF.VERSION_1_100, 'V_1_8': MF.VERSION_1_8, 'V_1_4': MF.VERSION_1_4}
oracle_idx = pickle.load(open(os.path.join(ref, 'indices.pkl'), 'rb'))
bad = 0
for ci, c in enumerate(CASES):
    for opt in OPTIMIZE:
        m = MeshData(f'case{ci}.mesh')
        m.set_bone_mapping([BoneData(i, f'bone{i}', Vector3(i, 0, 0), OgreQuaternion(1, 0, 0, 0), Vector3(1, 1, 1), 'bone0' if i else '', []) for i in range(5)])
        subs = []
        for si in range(2):
            d, ba, sk = make_case(c['seed'] * 10 + si, **{k: v for k, v in c.items() if k != 'seed'})
            s = SubMeshData(); s.index = si; s.submesh_name = f'sub{si}'; s.material = f'mat{si}'
            with contextlib.redirect_stdout(io.StringIO()):
                idx = s.set_vertex(optimize=opt, **d)
            if not np.array_equal(idx, oracle_idx[(ci, opt, si)]):
                print('index mismatch', ci, opt, si); bad += 1
            for name, off in sk: s.append_shapekey(name, off, idx)
            s.set_bone_assignments([BoneAssignmentData(*b) for b in ba], idx)
            subs.append(s)
        m.set_submeshes(subs)
        if ci != 4:
            m.set_linked_skeleton_name('case.skeleton')
        for vn, ver in VERSIONS.items():
            mine = MF.write_mesh(m._to_file(ver))
            theirs = open(os.path.join(ref, f'case{ci}_{int(opt)}_{vn}.mesh'), 'rb').read()
            if mine != theirs:
                bad += 1
                n = min(len(mine), len(theirs))
                diff = next((i for i in range(n) if mine[i] != theirs[i]), n)
                print(f'DIFF case{ci} opt{int(opt)} {vn}: sizes {len(mine)} vs {len(theirs)}, first diff at {diff}')
                print('   mine  ', mine[max(0, diff - 8):diff + 24].hex(' '))
                print('   oracle', theirs[max(0, diff - 8):diff + 24].hex(' '))
print('mismatches:', bad)
