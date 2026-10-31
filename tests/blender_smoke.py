"""Run inside Blender: blender -b --factory-startup --python blender_smoke.py -- <package parent dir> <out dir>"""
import os
import sys
import traceback

import bpy

argv = sys.argv[sys.argv.index('--') + 1:]
PKG_PARENT, OUT = argv[0], argv[1]
os.makedirs(OUT, exist_ok=True)
import addon_utils  # noqa: E402
# PKG_PARENT/addons/kenshi_io_tools, found through BLENDER_USER_SCRIPTS=PKG_PARENT
addon_utils.enable('kenshi_io_tools', default_set=True, handle_error=None)
assert 'kenshi_io_tools' in bpy.context.preferences.addons, 'add-on did not enable'

DATA = os.environ.get('KENSHI_DATA', 'C:/Program Files (x86)/Steam/steamapps/common/Kenshi/data').replace('\\', '/')
results = []


def check(name, cond, info=''):
    results.append((name, bool(cond), info))
    print(('PASS ' if cond else 'FAIL ') + name, info)


def reset():
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    for collection in (bpy.data.objects, bpy.data.meshes, bpy.data.armatures, bpy.data.actions,
                       bpy.data.materials):
        for item in list(collection):
            collection.remove(item)


def run(label, fn):
    try:
        fn()
    except Exception:
        check(label, False, traceback.format_exc())


def mesh_objects():
    return [o for o in bpy.data.objects if o.type == 'MESH']


def test_mesh_with_animation():
    reset()
    res = getattr(bpy.ops, 'import').kenshi_ogre_objects(filepath=f'{DATA}/animal/meshes/bull.mesh', import_animations=True)
    arm = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    meshes = mesh_objects()
    check('bull import finished', res == {'FINISHED'}, res)
    check('bull armature', len(arm) == 1 and len(arm[0].data.bones) == 37, [len(a.data.bones) for a in arm])
    check('bull mesh', len(meshes) == 1 and len(meshes[0].data.vertices) > 100, [len(m.data.vertices) for m in meshes])
    check('bull actions', len(bpy.data.actions) == 9, [a.name for a in bpy.data.actions])
    act = bpy.data.actions.get('run')
    fcurves = act.layers[0].strips[0].channelbags[0].fcurves if act else []
    check('bull fcurves', act is not None and len(fcurves) > 100, len(fcurves))
    check('bull weights', len(meshes[0].vertex_groups) > 10, len(meshes[0].vertex_groups))
    check('bull uv', len(meshes[0].data.uv_layers) == 1, len(meshes[0].data.uv_layers))

    # export everything back out and read it again
    for o in bpy.data.objects:
        o.select_set(o.type == 'MESH')
    bpy.context.view_layer.objects.active = meshes[0]
    arm[0].animation_data.action = act
    out_mesh = os.path.join(OUT, 'bull_export.mesh')
    res = getattr(bpy.ops, 'export').kenshi_ogre_objects(filepath=out_mesh, export_skeleton=True, export_animation=True,
                                                         export_version='V_1_8')
    check('bull export finished', res == {'FINISHED'} and os.path.isfile(out_mesh)
          and os.path.isfile(os.path.join(OUT, 'bull_export.skeleton')), res)
    reset()
    res = getattr(bpy.ops, 'import').kenshi_ogre_objects(filepath=out_mesh, import_animations=True)
    arm2 = [o for o in bpy.data.objects if o.type == 'ARMATURE']
    check('bull reimport', res == {'FINISHED'} and len(arm2) == 1 and len(arm2[0].data.bones) == 37
          and len(mesh_objects()) == 1, [len(a.data.bones) for a in arm2])
    check('bull reimport actions', len(bpy.data.actions) >= 1, [a.name for a in bpy.data.actions])


def test_shapekeys():
    reset()
    res = getattr(bpy.ops, 'import').kenshi_ogre_objects(filepath=f'{DATA}/character/meshes/bone/bone_female.mesh')
    m = mesh_objects()
    keys = m[0].data.shape_keys.key_blocks if m and m[0].data.shape_keys else []
    check('shape keys import', res == {'FINISHED'} and len(keys) == 36, len(keys))
    for o in bpy.data.objects:
        o.select_set(o.type == 'MESH')
    out = os.path.join(OUT, 'bone_female_export.mesh')
    res = getattr(bpy.ops, 'export').kenshi_ogre_objects(filepath=out, export_poses=True, export_colour=True,
                                                         tangent_format='ALL')
    check('shape keys export', res == {'FINISHED'} and os.path.isfile(out), res)
    reset()
    getattr(bpy.ops, 'import').kenshi_ogre_objects(filepath=out)
    m = mesh_objects()
    keys = m[0].data.shape_keys.key_blocks if m and m[0].data.shape_keys else []
    check('shape keys roundtrip', len(keys) == 36, len(keys))


def test_skeleton_file():
    reset()
    res = getattr(bpy.ops, 'import').kenshi_ogre_skeleton_objects(
        filepath=f'{DATA}/animal/meshes/bull.skeleton', import_animations=True, round_frames=True)
    check('skeleton import', res == {'FINISHED'} and len(bpy.data.actions) == 9, len(bpy.data.actions))
    for o in bpy.data.objects:
        o.select_set(o.type == 'ARMATURE')
    out = os.path.join(OUT, 'bull_only.skeleton')
    res = getattr(bpy.ops, 'export').kenshi_ogre_skeleton_objects(filepath=out, export_animation=True,
                                                                  is_visual_keying=True)
    check('skeleton export baked', res == {'FINISHED'} and os.path.getsize(out) > 1000, res)


def test_collision():
    reset()
    src = f'{DATA}/buildings/bunkerhouse/combos/moor27-trigger.xml'
    res = getattr(bpy.ops, 'import').kenshi_physx_objects(filepath=src)
    bodies = [o for o in bpy.data.objects if o.rigid_body]
    check('collision import', res == {'FINISHED'} and len(bodies) > 0,
          sorted({o.rigid_body.collision_shape for o in bodies}))
    out = os.path.join(OUT, 'collision_export.xml')
    res = getattr(bpy.ops, 'export').kenshi_physx_objects(filepath=out, objects='ALL')
    check('collision export', res == {'FINISHED'} and os.path.isfile(out) and 'cookedData' in open(out).read(), res)
    n = len(bodies)
    reset()
    getattr(bpy.ops, 'import').kenshi_physx_objects(filepath=out)
    check('collision roundtrip', len([o for o in bpy.data.objects if o.rigid_body]) == n, n)


for name in ('test_mesh_with_animation', 'test_shapekeys', 'test_skeleton_file', 'test_collision'):
    run(name, globals()[name])

print('SUMMARY', sum(ok for _, ok, _ in results), '/', len(results))
