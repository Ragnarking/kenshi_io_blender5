import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'addon'))
sys.path.insert(0, os.path.dirname(__file__))
from kenshi_blender_tool import _skeleton_file as SF
from kenshi_blender_tool import _skeleton, _math
from skel_export_cases import CASES, build_skeleton


class NewModule:
    BoneData = _skeleton.BoneData
    AnimationData = _skeleton.AnimationData
    Vector3 = _math.Vector3
    OgreQuaternion = _math.OgreQuaternion
    Matrix3 = _math.Matrix3


class Ser:
    def create_skeleton(self, name):
        return _skeleton.SkeletonData(name)


ref = sys.argv[1]
VERSIONS = {'V_Latest': SF.VERSION_1_8, 'V_1_8': SF.VERSION_1_8, 'V_1_0': SF.VERSION_1_0}
bad = 0
for ci, c in enumerate(CASES):
    for vn, ver in VERSIONS.items():
        sk = build_skeleton(NewModule, Ser(), c)
        mine = SF.write_skeleton(sk._to_file(ver))
        theirs = open(os.path.join(ref, f'skel{ci}_{vn}.skeleton'), 'rb').read()
        if mine != theirs:
            bad += 1
            n = min(len(mine), len(theirs))
            diff = next((i for i in range(n) if mine[i] != theirs[i]), n)
            print(f'DIFF skel{ci} {vn}: sizes {len(mine)} vs {len(theirs)}, first diff at {diff}')
            print('   mine  ', mine[max(0, diff - 12):diff + 24].hex(' '))
            print('   oracle', theirs[max(0, diff - 12):diff + 24].hex(' '))
            a = SF.read_skeleton(mine); b = SF.read_skeleton(theirs)
            for x, y in zip(a.animations, b.animations):
                for t, u in zip(x.tracks, y.tracks):
                    for f in ('times', 'rotations', 'translations', 'scales'):
                        d = np.abs(getattr(t, f) - getattr(u, f)).max() if getattr(t, f).shape == getattr(u, f).shape else 'shape'
                        if d == 'shape' or d > 1e-6: print('     track', t.bone, f, d)
                    break
                break
print('mismatches:', bad)
