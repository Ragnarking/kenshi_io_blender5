"""Pure Python replacement of the native Kenshi_blender_tool module.

The original module was a pybind11 binary built for one CPython version
(cp310 / cp311), so the add-on stopped loading once Blender moved to newer
Python releases. This package keeps the same classes and methods, implements
the Ogre mesh/skeleton formats in Python and reaches the bundled PhysX 2.8.4
libraries through ctypes, so it runs on any Blender Python.
"""
from ._math import Matrix3, OgreQuaternion, Radian, Vector3
from ._mesh import (BoneAssignmentData, GeometryData, MeshData, MeshVersion, OperationType,
                    SubMeshData, VertexGroupData)
from ._skeleton import (AnimationData, BlenderAnimationTrack, BoneData, SkeletonAnimationBlendMode,
                        SkeletonData, SkeletonVersion)
from ._serializer import KenshiObjectSerializer
from ._physx import CollisionMesh, KenshiPhysXSerializer

V_Latest = SkeletonVersion.V_Latest
V_1_0 = SkeletonVersion.V_1_0
V_1_8 = SkeletonVersion.V_1_8
V_1_10 = MeshVersion.V_1_10
V_1_7 = MeshVersion.V_1_7
V_1_4 = MeshVersion.V_1_4
V_Legacy = MeshVersion.V_Legacy
average = SkeletonAnimationBlendMode.average
cumulative = SkeletonAnimationBlendMode.cumulative
triangle_list = OperationType.triangle_list

__all__ = ['AnimationData', 'BlenderAnimationTrack', 'BoneAssignmentData', 'BoneData', 'CollisionMesh',
           'GeometryData', 'KenshiObjectSerializer', 'KenshiPhysXSerializer', 'Matrix3', 'MeshData',
           'MeshVersion', 'OgreQuaternion', 'OperationType', 'Radian', 'SkeletonAnimationBlendMode',
           'SkeletonData', 'SkeletonVersion', 'SubMeshData', 'Vector3', 'VertexGroupData']
