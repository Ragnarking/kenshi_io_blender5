"""Full chunk tree for Ogre-written meshes (lengths trusted)."""
import struct, sys
NEST = {0x3000: 1, 0x4000: None, 0x5000: 4, 0x5100: 0, 0x5200: 4, 0xA000: 0, 0xB000: 0, 0xC000: 0, 0xD000: 0}
def walk(b, pos, end, depth):
    while pos + 6 <= end:
        cid, ln = struct.unpack_from('<HI', b, pos)
        body = b[pos + 6:pos + ln]
        print('  ' * depth + f'{cid:04X} len={ln} {body[:40].hex(" ")}')
        if cid == 0x3000: walk(b, pos + 7, pos + ln, depth + 1)
        elif cid in (0x5000,): walk(b, pos + 10, pos + ln, depth + 1)
        elif cid == 0x5200: walk(b, pos + 10, pos + ln, depth + 1)
        elif cid in (0x5100, 0xA000, 0xC000, 0xD000, 0xB000): walk(b, pos + 6, pos + ln, depth + 1)
        elif cid == 0xC100:
            e = b.index(b'\n', pos + 6); walk(b, e + 1 + 3, pos + ln, depth + 1)
        elif cid == 0x4000:
            e = b.index(b'\n', pos + 6); shared = b[e + 1]; cnt, = struct.unpack_from('<I', b, e + 2); is32 = b[e + 6]
            p = e + 7 + cnt * (4 if is32 else 2); walk(b, p, pos + ln, depth + 1)
        pos += ln
b = open(sys.argv[1], 'rb').read(); e = b.index(b'\n'); print(b[2:e]); walk(b, e + 1, len(b), 0)
