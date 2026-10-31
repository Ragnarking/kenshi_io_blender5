"""Synthetic export inputs shared by oracle and new implementation."""
import numpy as np

def make_case(seed, n_verts=30, n_tris=40, tangent_format='TANGENT_4', colors=True, uv=True, bones=True, shapekeys=True, dup_loops=True):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n_verts, 3)).astype(np.float32)
    tris = np.array([rng.choice(n_verts, 3, replace=False) for _ in range(n_tris)], dtype=np.int32)
    vert_idx = tris.ravel().astype(np.int32)
    L = len(vert_idx)
    loop_idx = np.arange(L, dtype=np.int32)
    nrm = rng.normal(size=(L, 3)).astype(np.float32); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    if dup_loops:  # make loops of same vertex share attributes most of the time (smooth shading)
        for v in range(n_verts):
            ls = np.where(vert_idx == v)[0]
            if len(ls) > 1 and rng.random() < 0.7:
                nrm[ls] = nrm[ls[0]]
    uvs = rng.random((L, 2)).astype(np.float32)
    if dup_loops:
        for v in range(n_verts):
            ls = np.where(vert_idx == v)[0]
            if len(ls) > 1 and rng.random() < 0.8:
                uvs[ls] = uvs[ls[0]]
    tan = rng.normal(size=(L, 3)).astype(np.float32)
    sign = np.where(rng.random(L) < 0.5, -1.0, 1.0).astype(np.float32)
    bit = rng.normal(size=(L, 3)).astype(np.float32)
    col = rng.random((L, 4)).astype(np.float32)
    alp = rng.random((L, 4)).astype(np.float32)
    d = dict(nd_vert_indices=vert_idx, nd_loop_indices=loop_idx, nd_positions=pos, nd_normals=nrm,
             nd_tangents=tan, nd_bitangent_signs=sign, nd_bitangents=bit,
             nd_texcoords=uvs if uv else np.empty(2, np.float32),
             nd_colors=col if colors else np.empty(4, np.float32), nd_alphas=alp if colors else np.empty(4, np.float32),
             tangent_dimensions=4 if tangent_format in ('TANGENT_4', 'ALL', 'FLIPPED') else 3)
    if tangent_format == 'TANGENT_0':
        d['nd_tangents'] = np.empty(3, np.float32); d['nd_bitangent_signs'] = np.empty(1, np.float32); d['nd_bitangents'] = np.empty(3, np.float32)
    elif tangent_format == 'TANGENT_4':
        d['nd_bitangents'] = np.empty(3, np.float32)
    elif tangent_format == 'TANGENT_3':
        d['nd_bitangents'] = np.empty(3, np.float32)
    ba = []
    if bones:
        for v in range(n_verts):
            for b in rng.choice(5, rng.integers(1, 4), replace=False):
                ba.append((v, int(b), float(rng.random())))
    sk = []
    if shapekeys:
        for k in range(2):
            off = np.zeros((n_verts, 3), np.float32); m = rng.random(n_verts) < 0.4
            off[m] = rng.normal(size=(m.sum(), 3)) * 0.1
            sk.append((f'key{k}', off))
    return d, ba, sk

CASES = [
    dict(seed=1),
    dict(seed=2, tangent_format='TANGENT_0'),
    dict(seed=3, tangent_format='ALL'),
    dict(seed=4, tangent_format='TANGENT_3', colors=False),
    dict(seed=5, uv=False, tangent_format='TANGENT_0', colors=False, bones=False, shapekeys=False),
    dict(seed=6, n_verts=200, n_tris=300),
    dict(seed=7, dup_loops=False),
]
OPTIMIZE = [True, False]
