import os
import math
import sys

import bpy # type: ignore
from mathutils import Vector # type: ignore

# install blender with sudo snap install blender --classic
# run with blender -b -P src/datasets/modelnet40/render_views.py > /dev/null

def import_object(obj_path):
    clear_scene()
    _, ext = os.path.splitext(obj_path)
    ext = ext.lower()
    if ext == ".stl":
        # bpy.ops.import_mesh.stl(filepath=obj_path)
        bpy.ops.wm.stl_import(filepath=obj_path)
    else:
        raise RuntimeError(f"unexpected extension: {ext}")
    
def create_object(obj_path):
    import_object(obj_path)
    obj = bpy.context.selected_objects[0]
    
    size = get_obj_size(obj)
    size = max(size)
    inv_size = (1 / size, 1 / size, 1 / size)
    bpy.ops.transform.resize(value=inv_size)
    
    com = get_obj_center(obj)
    obj.location -= com
    
    return obj
    

def get_obj_center(obj):
    local_bbox_center = 0.125 * sum((Vector(b) for b in obj.bound_box), Vector())
    global_bbox_center = obj.matrix_world @ local_bbox_center
    
    return global_bbox_center

def get_obj_size(obj):
    bbox = obj.bound_box
    min_x, min_y, min_z = map(min, zip(*bbox))
    max_x, max_y, max_z = map(max, zip(*bbox))
    
    return Vector((max_x - min_x, max_y - min_y, max_z - min_z))
    
    
def clear_scene():
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()
    
    
def create_camera(scene):
    cam = bpy.data.cameras.new("Camera")
    cam.lens = 30
    
    cam_obj = bpy.data.objects.new("Camera", cam)
    cam_obj.location = (0, 0, 0)
    cam_obj.rotation_euler = (0, 0, 0)
    
    scene.camera = cam_obj
    return cam_obj

def update_camera(camera, theta, phi, focus_point=Vector((0.0, 0.0, 0.0)), distance=1):
    
    pos_x = distance * math.sin(theta) * math.cos(phi)
    pos_y = distance * math.sin(theta) * math.sin(phi)
    pos_z = distance * math.cos(theta)
    
    camera.location = Vector((pos_x, pos_y, pos_z))
    
    loc_camera = camera.location

    direction = focus_point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    camera.rotation_euler = rot_quat.to_euler()
    

def create_light(scene):
    light_data = bpy.data.lights.new(name="Light", type="SUN")
    light_data.energy = 5.0
    
    light_obj = bpy.data.objects.new(name="Light", object_data=light_data)
    light_obj.location = (0, 0, 10)
    scene.collection.objects.link(light_obj)
    
    bpy.context.view_layer.objects.active = light_obj


def set_render_settings(scene):
    render = scene.render
    
    render.engine = "BLENDER_WORKBENCH" # "BLENDER_WORKBENCH" # "BLENDER_EEVEE_NEXT" # "CYCLES"
    render.film_transparent = True
    
    scene.cycles.device = "GPU"
    
    render.resolution_x = 224
    render.resolution_y = 224
    
    render.filepath = "render.png"
    render.image_settings.file_format = "PNG"
    
    return render

def render_object(camera, render, obj_path, save_to=None):
    path_list = obj_path.split("/")
    if save_to is None:
        save_to = path_list[:-3] + ["render"]
        save_to = "/".join(save_to)
        
    obj_class = path_list[-3]
    obj_split = path_list[-2]
    obj_name = path_list[-1].split(".")[0]
    
    
    obj = create_object(obj_path)
    obj.select_set(True)

    for theta in (30, 90, 120):
        for phi in range(40, 360, 40):
            update_camera(camera, math.radians(theta), math.radians(phi))
            bpy.ops.view3d.camera_to_view_selected()
            render.filepath = f"{save_to}/{obj_class}/{obj_split}/{obj_name}/{theta}_{phi}.png"
            bpy.ops.render.render(write_still=True)


def prepare_scene():
    scene = bpy.context.scene
    create_light(scene)
    camera = create_camera(scene)
    scene.camera = camera
    render = set_render_settings(scene)
    
    return scene, camera, render
    
def debug(scene):
    print(scene.camera)

def main():
    models_dir = "./data/ModelNet40/stl_models"
    save_to = "./data/ModelNet40/renders"

    scene, camera, render = prepare_scene()

    # To render a single model
    file = models_dir + "/sofa/train/sofa_0157" + ".stl"
    render_object(camera, render, file, save_to)
    return
    
    for root, directory, files in os.walk(models_dir):
        category = root.split('/')[-2]
        
        for i, file in enumerate(files):
            render_object(camera, render, os.path.join(root, file), save_to)
            if (i + 1) % 20 == 0:
                print(f"rendered {i + 1} files from {'/'.join(root.split('/')[-2:])}", file=sys.stderr)
        print(f"rendered {root}", file=sys.stderr)


if __name__ == "__main__":
    main()