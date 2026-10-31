import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'addon'))
from oracle import K
from kenshi_blender_tool import _skeleton_file as SF
from config import KENSHI_DATA, LOG  # noqa: E402
out = sys.argv[1]
ser = K.KenshiObjectSerializer(LOG)
sk = ser.create_skeleton('d.skeleton')
sk.set_bones([K.BoneData(i, f'b{i}', K.Vector3(0, i, 0), K.OgreQuaternion(1, 0, 0, 0), K.Vector3(1, 1, 1), 'b0' if i else '', []) for i in range(3)])
a = K.AnimationData(); a.name = 'x'; a.length = 1.0
n = 12; t = np.arange(n, dtype=np.float32) / 11
I = K.Matrix3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
loc = np.zeros((3, n), np.float32); loc[0, :] = 0.5; loc[0, 8:] = 0.7   # constant runs
rot = np.zeros((4, n), np.float32); rot[0] = 1
scl = np.ones((3, n), np.float32)
a.append_animation_track(bone_name='b0', bone_matrix=I, nd_times=t, nd_locations=loc.astype(object), nd_rotations=rot.astype(object), nd_scales=scl.astype(object), use_scale=False)
loc2 = np.full((3, n), 0.0004, np.float32)  # near identity
a.append_animation_track(bone_name='b1', bone_matrix=I, nd_times=t, nd_locations=loc2.astype(object), nd_rotations=rot.astype(object), nd_scales=scl.astype(object), use_scale=False)
sk.add_animation(a)
ser.save_skeleton(sk, out, K.SkeletonVersion.V_Latest)
s = SF.read_skeleton(open(out, 'rb').read())
for tr in s.animations[0].tracks: print('bone', tr.bone, 'keys', len(tr.times), tr.times.round(3), tr.translations[:, 0].round(3))
