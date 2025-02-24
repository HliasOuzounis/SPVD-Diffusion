import os
import math
import sys
import json

import bpy  # type: ignore
from mathutils import Vector  # type: ignore

# install blender with sudo snap install blender --classic
# run with blender -b -P datasets/.../render_views.py > /dev/null

# SOMETHING IS WRONG WITH THE TEXTURES

def import_object(obj_path):
    clear_scene()
    _, ext = os.path.splitext(obj_path)
    ext = ext.lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=obj_path)
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
    bpy.ops.object.select_by_type(type="MESH")
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
    rot_quat = direction.to_track_quat("-Z", "Y")

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

    render.engine = (
        "BLENDER_WORKBENCH"  # "BLENDER_WORKBENCH" # "BLENDER_EEVEE_NEXT" # "CYCLES"
    )
    render.film_transparent = True

    scene.cycles.device = "GPU"

    render.resolution_x = 224
    render.resolution_y = 224

    render.filepath = "render.png"
    render.image_settings.file_format = "PNG"

    return render


def render_object(camera, render, obj_path, save_to):
    path_list = obj_path.split("/")
    obj_name = path_list[-3]
    obj_class = path_list[-4]

    obj = create_object(obj_path)
    obj.select_set(True)

    for theta in (30, 90, 120):
        for phi in range(40, 360, 40):
            update_camera(camera, math.radians(theta), math.radians(phi))
            bpy.ops.view3d.camera_to_view_selected()
            render.filepath = (
                f"{save_to}/{obj_class}/{obj_name}/{theta}_{phi}.png"
            )
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


synsetid_to_cate = {
    "02691156": "airplane",
    "02747177": "can",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02834778": "bicycle",
    "02843684": "birdhouse",
    "02858304": "boat",
    "02871439": "bookshelf",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02942699": "camera",
    "02946921": "tin_can",
    "02954340": "cap",
    "02958343": "car",
    "02992529": "cellphone",
    "03001627": "chair",
    "03046257": "clock",
    "03085013": "keyboard",
    "03207941": "dishwasher",
    "03211117": "monitor",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "speaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorcycle",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote_control",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04379243": "table",
    "04401088": "telephone",
    "04460130": "tower",
    "04468005": "train",
    "04530566": "vessel",
    "04554684": "washer",
}

def main():
    models_dir = "./data/ShapeNetCore/obj"
    save_to = "./data/ShapeNetCore/renders"

    scene, camera, render = prepare_scene()

    logging_info = {}

    already_completed = [x.name for x in os.scandir(save_to) if x.is_dir()]

    for code_name in os.scandir(models_dir):
        try:
            category = synsetid_to_cate[code_name.name]
        except KeyError:
            continue
        print(f"Rendering {category}...", file=sys.stderr)
        
        for i, model in enumerate(os.scandir(code_name.path), start=1):
            if i > 1000:
                break

            if code_name.name in already_completed:
                continue
            
            render_object(camera, render, os.path.join(model.path, "models/model_normalized.obj"), save_to)

            if i % 50 == 0:
                print(f"{i} models rendered", file=sys.stderr)
            

        logging_info[category] = i
    
        with open(os.path.join(save_to, "render_log.json"), "w") as log_file:
            json.dump(logging_info, log_file, indent=4)

if __name__ == "__main__":
    main()
