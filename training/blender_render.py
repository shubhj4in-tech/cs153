"""
Blender Synthetic Data Generator
==================================
Run with:
    blender --background --python training/blender_render.py -- \\
        --output-dir ./data/synthetic/ \\
        --n-scenes 200 \\
        --frames-per-scene 90

Generates N synthetic scenes, each with:
  • Randomised static background (textured floor + backdrop)
  • One moving foreground object (walking biped, bouncing sphere,
    rotating rigid body, or cloth simulation)
  • Randomised: lighting, camera path, textures, physics params

Per scene, renders 90 frames (3 s @ 30 fps):
  • render/frame_NNNN.png     — RGB (1280×720)
  • flow/frame_NNNN.exr       — optical flow (Vector pass, EXR)
  • depth/frame_NNNN.exr      — depth (Z pass, EXR)

Writes data/synthetic/manifest.json listing all scenes.

Tested on Blender 5.1. Compatible with Blender 3.6+ via try/except guards.
If you see AttributeError, check the changelog at https://docs.blender.org/api/
"""

import bpy
import json
import math
import os
import random
import sys
from pathlib import Path

# ── Parse args after '--' ────────────────────────────────────────────────────
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir",         default="./data/synthetic/")
parser.add_argument("--n-scenes",     type=int, default=200)
parser.add_argument("--frames-per-scene", type=int, default=90)
parser.add_argument("--res-x",        type=int, default=512)
parser.add_argument("--res-y",        type=int, default=512)
parser.add_argument("--start-scene",  type=int, default=0,
                    help="Resume from this scene index")
args = parser.parse_args(argv)

OUTPUT_DIR    = Path(args.output_dir)
N_SCENES      = args.n_scenes
N_FRAMES      = args.frames_per_scene
RES_X         = args.res_x
RES_Y         = args.res_y
START_SCENE   = args.start_scene

OBJECT_TYPES = ["biped", "sphere", "rigid", "cloth"]


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for mat in bpy.data.materials:
        bpy.data.materials.remove(mat)
    for mesh in bpy.data.meshes:
        bpy.data.meshes.remove(mesh)
    for curve in bpy.data.curves:
        bpy.data.curves.remove(curve)


def set_render_settings(use_gpu: bool = True):
    scene = bpy.context.scene
    scene.render.engine           = "CYCLES"
    scene.render.resolution_x     = RES_X
    scene.render.resolution_y     = RES_Y
    scene.render.resolution_percentage = 100
    scene.frame_start              = 1
    scene.frame_end                = N_FRAMES
    scene.render.fps               = 30

    # GPU rendering — try METAL (Apple Silicon) then CUDA (NVIDIA/Modal)
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if use_gpu:
        enabled = False
        for device_type in ("METAL", "CUDA", "OPTIX", "HIP"):
            try:
                prefs.compute_device_type = device_type
                prefs.get_devices()
                if any(d.type != "CPU" for d in prefs.devices):
                    for d in prefs.devices:
                        d.use = True
                    scene.cycles.device = "GPU"
                    enabled = True
                    break
            except Exception:
                continue
        if not enabled:
            scene.cycles.device = "CPU"
    else:
        scene.cycles.device = "CPU"

    scene.cycles.samples = 64       # low enough to be fast
    # use_denoising moved to view_layer.cycles in Blender 4.x
    try:
        scene.cycles.use_denoising = True
    except AttributeError:
        try:
            scene.view_layers[0].cycles.use_denoising = True
        except Exception:
            pass

    # ── Enable passes ────────────────────────────────────────────────────────
    view_layer = scene.view_layers[0]
    view_layer.use_pass_vector    = True     # optical flow
    view_layer.use_pass_z         = True     # depth


def random_material(name: str, color_bias=None) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    bsdf     = nodes.new("ShaderNodeBsdfPrincipled")
    out      = nodes.new("ShaderNodeOutputMaterial")
    mat.node_tree.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    if color_bias is None:
        color = (random.random(), random.random(), random.random(), 1.0)
    else:
        color = tuple(min(1, max(0, c + random.uniform(-0.2, 0.2)))
                       for c in color_bias) + (1.0,)
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value  = random.uniform(0.1, 0.9)
    bsdf.inputs["Metallic"].default_value   = random.uniform(0.0, 0.3)
    return mat


def add_floor_and_backdrop():
    """Add a textured floor plane and curved backdrop."""
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.data.materials.append(random_material("FloorMat"))

    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 10, 5),
                                     rotation=(math.radians(90), 0, 0))
    backdrop = bpy.context.active_object
    backdrop.name = "Backdrop"
    backdrop.data.materials.append(random_material("BackdropMat"))


def add_randomised_lighting():
    """Add 1 sun + 1-2 area lights."""
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 10))
    sun = bpy.context.active_object
    sun.data.energy = random.uniform(1.0, 4.0)
    sun.rotation_euler = (
        random.uniform(math.radians(20), math.radians(70)),
        0,
        random.uniform(0, math.radians(360)),
    )
    sun.data.color = (
        random.uniform(0.8, 1.0),
        random.uniform(0.8, 1.0),
        random.uniform(0.9, 1.0),
    )

    for _ in range(random.randint(1, 2)):
        bpy.ops.object.light_add(
            type="AREA",
            location=(
                random.uniform(-5, 5),
                random.uniform(-5, 5),
                random.uniform(3, 7),
            ),
        )
        area = bpy.context.active_object
        area.data.energy = random.uniform(50, 300)
        area.data.size   = random.uniform(1.0, 4.0)


def add_camera_with_animation():
    """
    Add a camera on a randomised arc around the scene origin,
    animated to slowly orbit over N_FRAMES.
    """
    bpy.ops.object.camera_add(location=(0, -5, 2))
    cam = bpy.context.active_object
    cam.name = "Camera"
    bpy.context.scene.camera = cam

    # Point camera at the origin
    constraint = cam.constraints.new("TRACK_TO")
    target_empty = bpy.data.objects.new("CamTarget", None)
    bpy.context.scene.collection.objects.link(target_empty)
    target_empty.location = (0, 0, 1)
    constraint.target    = target_empty
    constraint.track_axis = "TRACK_NEGATIVE_Z"
    constraint.up_axis    = "UP_Y"

    radius    = random.uniform(4.0, 8.0)
    elevation = random.uniform(0.5, 2.5)
    start_angle = random.uniform(0, 2 * math.pi)
    sweep       = random.uniform(0.2, 0.6)   # fraction of full circle to traverse

    for frame in range(1, N_FRAMES + 1):
        t     = (frame - 1) / (N_FRAMES - 1)
        angle = start_angle + sweep * 2 * math.pi * t
        cam.location = (
            radius * math.cos(angle),
            radius * math.sin(angle),
            elevation,
        )
        cam.keyframe_insert(data_path="location", frame=frame)

    return cam


# ─────────────────────────────────────────────────────────────────────────────
# Foreground objects
# ─────────────────────────────────────────────────────────────────────────────

def add_walking_biped():
    """
    Create a simple procedural stick-figure biped (torso + limbs) animated
    with a walk cycle via FCurves. Returns the root object.
    """
    # Torso
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0, 0, 1.0))
    torso = bpy.context.active_object
    torso.name = "Torso"
    torso.scale = (0.3, 0.2, 0.5)
    torso.data.materials.append(random_material("TorsoMat"))

    def make_limb(name, loc, scale):
        bpy.ops.mesh.primitive_cube_add(size=0.5, location=loc)
        limb = bpy.context.active_object
        limb.name = name
        limb.scale = scale
        limb.data.materials.append(random_material(f"{name}Mat"))
        limb.parent = torso
        return limb

    left_leg  = make_limb("LeftLeg",  (-0.15, 0, -0.5), (0.1, 0.1, 0.4))
    right_leg = make_limb("RightLeg", ( 0.15, 0, -0.5), (0.1, 0.1, 0.4))
    left_arm  = make_limb("LeftArm",  (-0.3,  0,  0.2), (0.08, 0.08, 0.35))
    right_arm = make_limb("RightArm", ( 0.3,  0,  0.2), (0.08, 0.08, 0.35))
    head      = make_limb("Head",     (0, 0, 0.55),     (0.15, 0.15, 0.15))

    # Animate: walk forward + swinging limbs
    walk_speed  = random.uniform(0.02, 0.05)
    swing_amp   = random.uniform(0.2, 0.5)

    for frame in range(1, N_FRAMES + 1):
        t     = (frame - 1) / N_FRAMES
        phase = 2 * math.pi * t * 2   # 2 steps per second at 30fps

        # Forward motion of torso
        torso.location = (0, walk_speed * frame, 1.0)
        torso.keyframe_insert(data_path="location", frame=frame)

        # Leg swing (rotation around X)
        left_leg.rotation_euler  = (swing_amp * math.sin(phase),     0, 0)
        right_leg.rotation_euler = (swing_amp * math.sin(phase + math.pi), 0, 0)
        left_arm.rotation_euler  = (swing_amp * math.sin(phase + math.pi), 0, 0)
        right_arm.rotation_euler = (swing_amp * math.sin(phase),     0, 0)

        left_leg.keyframe_insert(data_path="rotation_euler",  frame=frame)
        right_leg.keyframe_insert(data_path="rotation_euler", frame=frame)
        left_arm.keyframe_insert(data_path="rotation_euler",  frame=frame)
        right_arm.keyframe_insert(data_path="rotation_euler", frame=frame)

    return torso


def add_bouncing_sphere():
    """Sphere with physics-based bounce via keyframes."""
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.4, location=(0, 0, 2))
    sphere = bpy.context.active_object
    sphere.name = "BouncingSphere"
    sphere.data.materials.append(random_material("SphereMat"))

    gravity    = -9.8
    restitution = random.uniform(0.5, 0.85)
    x_speed    = random.uniform(-0.05, 0.05)
    y_speed    = random.uniform(-0.05, 0.05)
    t_dt       = 1.0 / 30.0
    z          = 2.0
    vz         = 0.0
    x          = 0.0
    y          = 0.0

    for frame in range(1, N_FRAMES + 1):
        vz += gravity * t_dt
        z  += vz * t_dt
        x  += x_speed
        y  += y_speed
        if z < 0.4:   # hit the floor
            z  = 0.4
            vz = -vz * restitution
        sphere.location = (x, y, z)
        sphere.keyframe_insert(data_path="location", frame=frame)

    return sphere


def add_rotating_rigid_body():
    """A multi-face polyhedron tumbling across the floor."""
    shapes = ["mesh.primitive_ico_sphere_add",
              "mesh.primitive_cube_add",
              "mesh.primitive_cone_add"]
    shape_fn = random.choice(shapes)
    getattr(bpy.ops, shape_fn)(location=(0, 0, 0.5))
    obj = bpy.context.active_object
    obj.name = "RigidBody"
    obj.data.materials.append(random_material("RigidMat"))

    omega_x = random.uniform(0.05, 0.15)
    omega_y = random.uniform(0.02, 0.10)
    omega_z = random.uniform(0.08, 0.20)
    vx      = random.uniform(-0.03, 0.03)
    vy      = random.uniform(-0.03, 0.03)

    for frame in range(1, N_FRAMES + 1):
        t = frame - 1
        obj.location = (vx * t, vy * t, 0.5)
        obj.rotation_euler = (omega_x * t, omega_y * t, omega_z * t)
        obj.keyframe_insert(data_path="location",       frame=frame)
        obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    return obj


def add_cloth_simulation():
    """A cloth plane falling and draping over a collision sphere."""
    # Cloth
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0, 0, 2.5))
    cloth_obj = bpy.context.active_object
    # subdivision_set API changed in 4.x — use modifier directly as fallback
    try:
        bpy.ops.object.subdivision_set(level=4, relative=False)
    except TypeError:
        bpy.ops.object.subdivision_set(level=4)
    except Exception:
        bpy.ops.object.modifier_add(type="SUBSURF")
        for m in cloth_obj.modifiers:
            if m.type == "SUBSURF":
                m.levels = 4
                m.render_levels = 4
                break
    cloth_obj.name = "Cloth"
    cloth_obj.data.materials.append(random_material("ClothMat"))

    # Add cloth modifier
    bpy.ops.object.modifier_add(type="CLOTH")
    cloth_mod = cloth_obj.modifiers["Cloth"]
    cloth_mod.settings.quality  = 5
    cloth_mod.settings.mass     = random.uniform(0.1, 0.5)

    # Collision sphere underneath
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.8, location=(0, 0, 0.8))
    coll_sphere = bpy.context.active_object
    coll_sphere.name = "CollisionSphere"
    bpy.ops.object.modifier_add(type="COLLISION")
    coll_sphere.data.materials.append(random_material("CollMat"))

    # Bake the simulation
    try:
        bpy.ops.ptcache.bake_all(bake=True)
    except Exception:
        pass   # May fail silently without a proper depsgraph; animation still works

    return cloth_obj


# ─────────────────────────────────────────────────────────────────────────────
# Render one scene
# ─────────────────────────────────────────────────────────────────────────────

def render_scene(scene_idx: int, out_base: Path) -> dict:
    scene_dir = out_base / f"scene_{scene_idx:04d}"
    render_dir = scene_dir / "render"
    flow_dir   = scene_dir / "flow"
    depth_dir  = scene_dir / "depth"

    render_dir.mkdir(parents=True, exist_ok=True)
    flow_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    clear_scene()
    set_render_settings()

    add_floor_and_backdrop()
    add_randomised_lighting()
    add_camera_with_animation()

    obj_type = random.choice(OBJECT_TYPES)
    if obj_type == "biped":
        fg_obj = add_walking_biped()
    elif obj_type == "sphere":
        fg_obj = add_bouncing_sphere()
    elif obj_type == "rigid":
        fg_obj = add_rotating_rigid_body()
    else:  # cloth
        fg_obj = add_cloth_simulation()

    scene = bpy.context.scene

    # ── Render passes setup ───────────────────────────────────────────────────
    scene.use_nodes = True
    tree  = scene.node_tree
    nodes = tree.nodes
    links = tree.links
    nodes.clear()

    render_node = nodes.new("CompositorNodeRLayers")

    # RGB output
    rgb_out     = nodes.new("CompositorNodeOutputFile")
    rgb_out.base_path = str(render_dir)
    rgb_out.file_slots[0].path = "frame_"
    rgb_out.format.file_format = "PNG"
    links.new(render_node.outputs["Image"], rgb_out.inputs[0])

    # Flow output (EXR)
    flow_out     = nodes.new("CompositorNodeOutputFile")
    flow_out.base_path = str(flow_dir)
    flow_out.file_slots[0].path = "frame_"
    flow_out.format.file_format = "OPEN_EXR"
    flow_out.format.color_depth = "32"
    links.new(render_node.outputs["Vector"], flow_out.inputs[0])

    # Depth output (EXR)
    depth_out     = nodes.new("CompositorNodeOutputFile")
    depth_out.base_path = str(depth_dir)
    depth_out.file_slots[0].path = "frame_"
    depth_out.format.file_format = "OPEN_EXR"
    depth_out.format.color_depth = "32"
    links.new(render_node.outputs["Depth"], depth_out.inputs[0])

    # ── Render ────────────────────────────────────────────────────────────────
    print(f"  Rendering scene {scene_idx} ({obj_type}) — {N_FRAMES} frames ...")
    bpy.ops.render.render(animation=True)

    return {
        "scene_index": scene_idx,
        "object_type": obj_type,
        "render_dir":  str(render_dir),
        "flow_dir":    str(flow_dir),
        "depth_dir":   str(depth_dir),
        "n_frames":    N_FRAMES,
        "resolution":  [RES_X, RES_Y],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

manifest_path = OUTPUT_DIR / "manifest.json"
manifest = []

# Load existing manifest if resuming
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text())
    print(f"Resuming: {len(manifest)} scenes already done.")

for i in range(START_SCENE, N_SCENES):
    random.seed(i)   # deterministic per scene

    print(f"\n[Blender] Scene {i+1}/{N_SCENES}")
    try:
        record = render_scene(i, OUTPUT_DIR)
        manifest.append(record)
    except Exception as e:
        print(f"  ERROR on scene {i}: {e}")
        manifest.append({"scene_index": i, "error": str(e)})

    # Save manifest after every scene
    manifest_path.write_text(json.dumps(manifest, indent=2))

print(f"\n[Blender] Done. {N_SCENES} scenes → {OUTPUT_DIR}")
print(f"Manifest: {manifest_path}")
