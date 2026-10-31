"""Small Ogre style math types with the interface of the original native module."""
import math

import numpy as np


class Radian:
    __hash__ = None

    def __init__(self, r: float = 0.0):
        self._r = float(r)

    def valueRadians(self) -> float:
        return self._r

    def __add__(self, o): return Radian(self._r + o._r)
    def __sub__(self, o): return Radian(self._r - o._r)
    def __neg__(self): return Radian(-self._r)
    def __mul__(self, o): return Radian(self._r * (o._r if isinstance(o, Radian) else o))
    def __truediv__(self, o): return Radian(self._r / o)
    def __iadd__(self, o): self._r += o._r; return self
    def __isub__(self, o): self._r -= o._r; return self
    def __imul__(self, o): self._r *= o; return self
    def __itruediv__(self, o): self._r /= o; return self
    def __eq__(self, o): return isinstance(o, Radian) and self._r == o._r
    def __ne__(self, o): return not self == o
    def __lt__(self, o): return self._r < o._r
    def __le__(self, o): return self._r <= o._r
    def __gt__(self, o): return self._r > o._r
    def __ge__(self, o): return self._r >= o._r
    def __repr__(self): return f'Radian({self._r})'


class Vector3:
    __hash__ = None

    def __init__(self, x=0.0, y=None, z=None):
        if y is None and z is None:
            y = z = x if not isinstance(x, (list, tuple)) else None
            if isinstance(x, (list, tuple)):
                x, y, z = x
        self.x, self.y, self.z = float(x), float(y if y is not None else 0.0), float(z if z is not None else 0.0)

    def _t(self):
        return (self.x, self.y, self.z)

    @staticmethod
    def _o(o):
        return o._t() if isinstance(o, Vector3) else (o, o, o)

    def _op(self, o, f):
        a, b = self._t(), self._o(o)
        return Vector3(f(a[0], b[0]), f(a[1], b[1]), f(a[2], b[2]))

    def __add__(self, o): return self._op(o, lambda a, b: a + b)
    def __radd__(self, o): return self._op(o, lambda a, b: b + a)
    def __sub__(self, o): return self._op(o, lambda a, b: a - b)
    def __rsub__(self, o): return self._op(o, lambda a, b: b - a)
    def __mul__(self, o): return self._op(o, lambda a, b: a * b)
    def __rmul__(self, o): return self._op(o, lambda a, b: b * a)
    def __truediv__(self, o): return self._op(o, lambda a, b: a / b)
    def __rtruediv__(self, o): return self._op(o, lambda a, b: b / a)
    def __neg__(self): return Vector3(-self.x, -self.y, -self.z)

    def _assign(self, v):
        self.x, self.y, self.z = v.x, v.y, v.z
        return self

    def __iadd__(self, o): return self._assign(self + o)
    def __isub__(self, o): return self._assign(self - o)
    def __imul__(self, o): return self._assign(self * o)
    def __itruediv__(self, o): return self._assign(self / o)

    def __eq__(self, o): return isinstance(o, Vector3) and self._t() == o._t()
    def __ne__(self, o): return not self == o
    def __lt__(self, o): return self.x < o.x and self.y < o.y and self.z < o.z
    def __gt__(self, o): return self.x > o.x and self.y > o.y and self.z > o.z

    def dotProduct(self, o) -> float:
        return self.x * o.x + self.y * o.y + self.z * o.z

    def __repr__(self): return f'Vector3({self.x}, {self.y}, {self.z})'


class OgreQuaternion:
    __hash__ = None

    def __init__(self, *args, **kwargs):
        if len(args) == 2 and isinstance(args[0], Radian):
            angle, axis = args
            half = 0.5 * angle.valueRadians()
            s = math.sin(half)
            self.w, self.x, self.y, self.z = math.cos(half), s * axis.x, s * axis.y, s * axis.z
            return
        if 'angle' in kwargs:
            self.__init__(kwargs['angle'], kwargs['axis'])
            return
        values = dict(zip(('w', 'x', 'y', 'z'), args))
        values.update(kwargs)
        self.w = float(values.get('w', 1.0))
        self.x = float(values.get('x', 0.0))
        self.y = float(values.get('y', 0.0))
        self.z = float(values.get('z', 0.0))

    def _t(self):
        return (self.w, self.x, self.y, self.z)

    def __add__(self, o): return OgreQuaternion(*(a + b for a, b in zip(self._t(), o._t())))
    def __sub__(self, o): return OgreQuaternion(*(a - b for a, b in zip(self._t(), o._t())))
    def __neg__(self): return OgreQuaternion(*(-a for a in self._t()))

    def __mul__(self, o):
        if isinstance(o, OgreQuaternion):
            w1, x1, y1, z1 = self._t()
            w2, x2, y2, z2 = o._t()
            return OgreQuaternion(
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
                w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2)
        if isinstance(o, Vector3):
            m = quat_to_matrix(self._t())
            v = m @ np.array(o._t())
            return Vector3(*v)
        return OgreQuaternion(*(a * o for a in self._t()))

    def __eq__(self, o): return isinstance(o, OgreQuaternion) and self._t() == o._t()
    def __ne__(self, o): return not self == o

    def Dot(self, o) -> float:
        return sum(a * b for a, b in zip(self._t(), o._t()))

    def normalise(self) -> float:
        length = math.sqrt(self.Dot(self))
        if length > 0:
            self.w, self.x, self.y, self.z = (a / length for a in self._t())
        return length

    def to_angle_axis(self):
        """Same conventions as Ogre::Quaternion::ToAngleAxis."""
        sqr_length = self.x * self.x + self.y * self.y + self.z * self.z
        if sqr_length > 0.0:
            angle = 2.0 * math.acos(max(-1.0, min(1.0, self.w)))
            inv = 1.0 / math.sqrt(sqr_length)
            return Radian(angle), Vector3(self.x * inv, self.y * inv, self.z * inv)
        return Radian(0.0), Vector3(1.0, 0.0, 0.0)

    def __repr__(self): return f'OgreQuaternion({self.w}, {self.x}, {self.y}, {self.z})'


class Matrix3:
    __hash__ = None

    def __init__(self, *args):
        if len(args) == 0:
            self.m = np.zeros((3, 3))
        elif len(args) == 1 and isinstance(args[0], Matrix3):
            self.m = args[0].m.copy()
        elif len(args) == 1:
            self.m = np.array(args[0], dtype=np.float64).reshape(3, 3)
        else:
            self.m = np.array(args, dtype=np.float64).reshape(3, 3)

    def __getitem__(self, i):
        return self.m[i]

    def __add__(self, o): return Matrix3(self.m + o.m)
    def __sub__(self, o): return Matrix3(self.m - o.m)
    def __neg__(self): return Matrix3(-self.m)

    def __mul__(self, o):
        if isinstance(o, Matrix3):
            return Matrix3(self.m @ o.m)
        if isinstance(o, Vector3):
            return Vector3(*(self.m @ np.array(o._t())))
        return Matrix3(self.m * o)

    def __eq__(self, o): return isinstance(o, Matrix3) and bool(np.all(self.m == o.m))
    def __ne__(self, o): return not self == o
    def __repr__(self): return f'Matrix3({self.m.tolist()})'


# ------------------------------------------------- numpy helpers (w, x, y, z order)

def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quats_to_matrices(q: np.ndarray) -> np.ndarray:
    """(n, 4) wxyz -> (n, 3, 3)"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    m = np.empty((len(q), 3, 3))
    m[:, 0, 0] = 1 - 2 * (y * y + z * z)
    m[:, 0, 1] = 2 * (x * y - w * z)
    m[:, 0, 2] = 2 * (x * z + w * y)
    m[:, 1, 0] = 2 * (x * y + w * z)
    m[:, 1, 1] = 1 - 2 * (x * x + z * z)
    m[:, 1, 2] = 2 * (y * z - w * x)
    m[:, 2, 0] = 2 * (x * z - w * y)
    m[:, 2, 1] = 2 * (y * z + w * x)
    m[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def matrices_to_quats(m: np.ndarray) -> np.ndarray:
    """(n, 3, 3) -> (n, 4) wxyz, same algorithm as Ogre::Quaternion::FromRotationMatrix."""
    out = np.empty((len(m), 4))
    for i, k in enumerate(m):
        out[i] = matrix_to_quat(k)
    return out


def matrix_to_quat(k: np.ndarray):
    trace = k[0, 0] + k[1, 1] + k[2, 2]
    if trace > 0.0:
        root = math.sqrt(trace + 1.0)
        w = 0.5 * root
        root = 0.5 / root
        return (w, (k[2, 1] - k[1, 2]) * root, (k[0, 2] - k[2, 0]) * root, (k[1, 0] - k[0, 1]) * root)
    nxt = (1, 2, 0)
    i = 0
    if k[1, 1] > k[0, 0]:
        i = 1
    if k[2, 2] > k[i, i]:
        i = 2
    j = nxt[i]
    kk = nxt[j]
    root = math.sqrt(k[i, i] - k[j, j] - k[kk, kk] + 1.0)
    q = [0.0, 0.0, 0.0]
    q[i] = 0.5 * root
    root = 0.5 / root
    w = (k[kk, j] - k[j, kk]) * root
    q[j] = (k[j, i] + k[i, j]) * root
    q[kk] = (k[kk, i] + k[i, kk]) * root
    return (w, q[0], q[1], q[2])
