"""KenshiObjectSerializer: loads and saves .mesh / .skeleton files."""
import os
import time
from typing import List

from . import _mesh_file as MF
from . import _skeleton_file as SF
from ._mesh import MeshData, MeshVersion
from ._skeleton import SkeletonData, SkeletonVersion

_MESH_VERSIONS = {
    MeshVersion.V_Latest: MF.VERSION_1_100,
    MeshVersion.V_1_10: MF.VERSION_1_100,
    MeshVersion.V_1_8: MF.VERSION_1_8,
    MeshVersion.V_1_7: MF.VERSION_1_41,
    MeshVersion.V_1_4: MF.VERSION_1_4,
}
_SKELETON_VERSIONS = {
    SkeletonVersion.V_Latest: SF.VERSION_1_8,
    SkeletonVersion.V_1_8: SF.VERSION_1_8,
    SkeletonVersion.V_1_0: SF.VERSION_1_0,
}


class KenshiObjectSerializer:
    def __init__(self, logfile: str = 'Kenshi_io_OGRE.log'):
        self._logfile = logfile
        self._locations: List[str] = []

    def _log(self, message: str):
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._logfile)), exist_ok=True)
            with open(self._logfile, 'a', encoding='utf-8') as f:
                f.write(f'{time.strftime("%H:%M:%S")}: {message}\n')
        except OSError:
            pass

    def add_resource_location(self, path: str):
        path = os.path.abspath(path)
        if path not in self._locations:
            self._locations.append(path)

    def _find(self, name: str) -> str:
        if os.path.isabs(name) and os.path.isfile(name):
            return name
        for location in self._locations:
            candidate = os.path.join(location, name)
            if os.path.isfile(candidate):
                return candidate
        if os.path.isfile(name):
            return os.path.abspath(name)
        raise FileNotFoundError(f'Cannot locate resource {name}')

    def create_mesh(self, name: str) -> MeshData:
        mesh = MeshData(name)
        mesh._serializer = self
        return mesh

    def create_skeleton(self, name: str) -> SkeletonData:
        return SkeletonData(name)

    def load_mesh(self, name: str) -> MeshData:
        path = self._find(name)
        self._log(f'Loading mesh {path}')
        with open(path, 'rb') as f:
            mesh = MeshData._from_file(path, MF.read_mesh(f.read()), self)
        self.add_resource_location(os.path.dirname(path))
        return mesh

    def load_skeleton(self, name: str) -> SkeletonData:
        path = self._find(name)
        self._log(f'Loading skeleton {path}')
        with open(path, 'rb') as f:
            return SkeletonData._from_file(path, SF.read_skeleton(f.read()))

    def save_mesh(self, mesh: MeshData, path: str, version=MeshVersion.V_Latest):
        if version not in _MESH_VERSIONS:
            raise ValueError(f'Unsupported mesh version {version!r}')
        data = MF.write_mesh(mesh._to_file(_MESH_VERSIONS[version]))
        with open(path, 'wb') as f:
            f.write(data)
        self._log(f'Saved mesh {path} ({_MESH_VERSIONS[version]})')

    def save_skeleton(self, skeleton: SkeletonData, path: str, version=SkeletonVersion.V_Latest):
        if version not in _SKELETON_VERSIONS:
            raise ValueError(f'Unsupported skeleton version {version!r}')
        data = SF.write_skeleton(skeleton._to_file(_SKELETON_VERSIONS[version]))
        with open(path, 'wb') as f:
            f.write(data)
        self._log(f'Saved skeleton {path} ({_SKELETON_VERSIONS[version]})')
