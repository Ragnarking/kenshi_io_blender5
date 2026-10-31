"""Little-endian binary helpers shared by the Ogre mesh and skeleton serializers."""
import struct

import numpy as np

CHUNK_HEADER_SIZE = 6  # unsigned short id + unsigned int length


class Reader:
    def __init__(self, data: bytes, pos: int = 0):
        self.data = data
        self.pos = pos

    def eof(self, end=None) -> bool:
        return self.pos >= (len(self.data) if end is None else end)

    def bytes(self, n: int) -> bytes:
        b = self.data[self.pos:self.pos + n]
        if len(b) != n:
            raise EOFError('Unexpected end of file')
        self.pos += n
        return b

    def unpack(self, fmt: str):
        s = struct.calcsize('<' + fmt)
        v = struct.unpack_from('<' + fmt, self.data, self.pos)
        self.pos += s
        return v

    def bool(self) -> bool:
        return self.bytes(1) != b'\x00'

    def u16(self) -> int:
        return self.unpack('H')[0]

    def u32(self) -> int:
        return self.unpack('I')[0]

    def f32(self) -> float:
        return self.unpack('f')[0]

    def floats(self, n: int):
        return self.unpack(f'{n}f')

    def array(self, dtype, count: int) -> np.ndarray:
        dt = np.dtype(dtype).newbyteorder('<')
        a = np.frombuffer(self.data, dtype=dt, count=count, offset=self.pos)
        self.pos += a.nbytes
        return a.astype(dt.newbyteorder('='), copy=True)

    def string(self) -> bytes:
        """Ogre strings are terminated by a newline."""
        end = self.data.find(b'\n', self.pos)
        if end < 0:
            end = len(self.data)
        s = self.data[self.pos:end]
        self.pos = end + 1
        return s

    def chunk(self):
        cid, length = struct.unpack_from('<HI', self.data, self.pos)
        start = self.pos
        self.pos += CHUNK_HEADER_SIZE
        return cid, start, start + length

    def peek_chunk_id(self):
        if self.pos + 2 > len(self.data):
            return None
        return struct.unpack_from('<H', self.data, self.pos)[0]


class Writer:
    def __init__(self):
        self.parts = []
        self._stack = []

    def raw(self, b: bytes):
        self.parts.append(bytes(b))

    def pack(self, fmt: str, *values):
        self.parts.append(struct.pack('<' + fmt, *values))

    def bool(self, v: bool):
        self.parts.append(b'\x01' if v else b'\x00')

    def u16(self, v: int):
        self.pack('H', v)

    def u32(self, v: int):
        self.pack('I', v)

    def f32(self, v: float):
        self.pack('f', v)

    def floats(self, *values):
        self.pack(f'{len(values)}f', *values)

    def array(self, a: np.ndarray, dtype):
        self.parts.append(np.ascontiguousarray(a, dtype=np.dtype(dtype).newbyteorder('<')).tobytes())

    def string(self, s):
        if isinstance(s, str):
            s = s.encode('utf-8')
        self.parts.append(s + b'\n')

    def begin_chunk(self, cid: int):
        self._stack.append((len(self.parts), cid))
        self.parts.append(None)  # placeholder for the header

    def end_chunk(self, length: int = None):
        """`length` overrides the stored size, for chunks Ogre sizes differently from their content."""
        index, cid = self._stack.pop()
        if length is None:
            length = CHUNK_HEADER_SIZE + sum(len(p) for p in self.parts[index + 1:])
        self.parts[index] = struct.pack('<HI', cid, length)

    def getvalue(self) -> bytes:
        assert not self._stack
        return b''.join(self.parts)
