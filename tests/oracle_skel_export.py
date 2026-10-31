import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from oracle import K
from skel_export_cases import CASES, build_skeleton
from config import KENSHI_DATA, LOG  # noqa: E402
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for ci, c in enumerate(CASES):
    for vn in ('V_Latest', 'V_1_8', 'V_1_0'):
        ser = K.KenshiObjectSerializer(LOG)
        sk = build_skeleton(K, ser, c)
        ser.save_skeleton(sk, os.path.join(out, f'skel{ci}_{vn}.skeleton'), getattr(K.SkeletonVersion, vn))
        del sk, ser
print('ok')
