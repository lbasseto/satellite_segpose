import os
import bpy
import sys
import random
import math
from math import pi
from mathutils import Matrix, Vector

# Build intrinsic camera parameters from Blender camera data
#
def get_calibration_matrix_K_from_blender(camera, render):
    assert (camera.type == 'CAMERA')
    K = camera.calc_matrix_camera(
        render.resolution_x,
        render.resolution_y,
        render.pixel_aspect_x,
        render.pixel_aspect_y,
    )
    assert (K[0][1] == 0)  # assert no skew
    assert (K[3][2] == -1)
    fx = K[0][0] * render.resolution_x / 2.0
    fy = K[1][1] * render.resolution_y / 2.0
    du = render.resolution_x / 2.0
    dv = render.resolution_y / 2.0

    # now K is the normal 3x3 matrix
    K = Matrix(((fx, 0, du), (0, fy, dv), (0, 0, 1)))

    # print('K:', K)
    return K

# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_4x4_RT_matrix_from_blender(camera):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
         (0, -1, 0),
         (0, 0, -1)))
    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    R_world2bcam = camera.rotation_euler.to_matrix().transposed()
    T_world2bcam = -1*R_world2bcam * camera.location
    #
    # Use matrix_world instead to account for all constraints
    # location, rotation = camera.matrix_world.decompose()[0:2]
    # R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    # T_world2bcam = -1*R_world2bcam * location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv*R_world2bcam
    T_world2cv = R_bcam2cv*T_world2bcam

    # put into 4x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],),
        (0,0,0,1)))
    return RT

def GetPose(camera, obj, render = bpy.context.scene.render):
    """
    Given a camera, a object and the render settings,
    return the pose of the object also the intrinsics K of the camera
    """
    K = get_calibration_matrix_K_from_blender(camera, render)
    RTc = get_4x4_RT_matrix_from_blender(camera)

    # get the object pose w.r.t. the world origin
    R = obj.rotation_euler.to_matrix()
    T = Matrix.Translation(obj.location)
    RTo = T * R.to_4x4()
    
    pose = RTc * RTo

    # print('pose:', pose)
    # print('K:', K)
    return pose, K

def WriteMatrix(fileName, mat, dim):
    with open(fileName, 'w') as outfile:
        for i in range(dim):
            for j in range(dim):
                outfile.write("%f\t" % mat[i][j])
            outfile.write('\n')
        outfile.close()

def MakeCameraLookAt(obj_camera, obj_target):
    # loc_object = obj_target.matrix_world.to_translation()
    # loc_camera = obj_camera.matrix_world.to_translation()
    loc_object = obj_target.location
    loc_camera = obj_camera.location
    
    direction = loc_object - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()
    
    # debug
    #print("camera pose:")
    #print(obj_camera.location)
    #print(obj_camera.rotation_euler)
    
def Render(outputDir, camObject, lampObj, imObject, rd):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir) # recursive

    imObject.location = (0, 0, 0)
    imObject.rotation_euler = (0, 0, 0)
    
    idx = 0
    scale = 1 # camera distance to the origin
    for camTh in range(1, 90, 2):
        for objTh in range(-180, 180, 2):
            camx = scale * math.cos(camTh * pi / 180)
            camy = 0
            camz = scale * math.sin(camTh * pi / 180)

            imObject.rotation_euler = (0, 0, objTh * pi / 180)
            # imObject.rotation_euler = (pi, 0, objTh * pi / 180)

            camObject.location =(camx, camy, camz)
            MakeCameraLookAt(camObject, imObject)

            # set lamp direction according to the viewpoint
            x = random.normalvariate(0, 1)
            y = random.normalvariate(0, 1)
            z = random.normalvariate(0, 1)
            if z < 0:
                z = -z
            norm = math.sqrt(x * x + y * y + z * z)
            x = scale * x / norm
            y = scale * y / norm
            z = scale * z / norm
            lampObj.location = (x,y,z)
            MakeCameraLookAt(lampObj, imObject)

            pose, K = GetPose(camObject, imObject)
            outpath = outputDir + ("/%05d" % idx)
            WriteMatrix(outpath + '.txt', pose, 4)

            rd.filepath = outpath + '.png'
            bpy.ops.render.render(write_still=True)

            idx += 1
        
    
def SetRenderingEnv(modelFile):
    
    # default camera parameters
    camobj = bpy.data.objects['Camera']
    campos = camobj.location
    camrot = camobj.rotation_euler
    camlens = bpy.data.cameras['Camera'].lens
    # print(campos, camrot, camlens)

    # delete all except the camera
    bpy.ops.object.select_all(action='SELECT')
    camobj.select = False
    bpy.ops.object.delete()

    # -- Place camera
    #bpy.ops.object.camera_add(view_align = False, location=(0,0,0), rotation=(0,0,0))
    #bpy.context.scene.camera = bpy.context.object # Set camera using the just created camera
    #camobj = bpy.context.object
    
    #
    #camobj.location = (1.5, -1.5, 1)
    #camobj.rotation_euler = (0, 0, 0)
    bpy.data.cameras[0].lens = 50

    bpy.ops.import_mesh.ply(filepath=modelFile)
    #bpy.ops.import_mesh.stl(filepath='kb1.stl')
    importedObject = bpy.context.object # Get the just imported object.

    # -- Place lights
    bpy.ops.object.lamp_add(
        type='SUN',
        view_align=False, rotation=(0, 0, 0),
        layers=tuple(i == 0 for i in range(20))
    )
    lampObj = bpy.context.object # Get the just created lamp.
    lampObj.data.energy = 0.5  # set lamp strength

    bpy.ops.object.select_all(action='DESELECT')
    for ob in bpy.data.objects:
        print(ob.type, ob.name)
        if ob.type == 'MESH':
            bpy.context.scene.objects.active = ob
            ob.select = True
            mat = bpy.data.materials.new('material_1')
            ob.active_material = mat
            mat.use_vertex_color_paint = True
        elif ob.type == 'CAMERA':
            ob.data.clip_end = 1000000
            ob.data.clip_start = 0.01
            ob.select = False
        else:
            ob.select = False
            
    #-- set the rendering options
    rd = bpy.context.scene.render
    rd.resolution_x = 640
    rd.resolution_y = 480
    rd.alpha_mode = 'TRANSPARENT'
    bpy.data.worlds["World"].light_settings.use_ambient_occlusion = True
    rd.resolution_percentage=100

    # for area in bpy.context.screen.areas:
    #     if area.type == 'VIEW_3D':
    #         ctx = bpy.context.copy()
    #         ctx['area'] = area
    #         ctx['region'] = area.regions[-1]
    #         bpy.ops.view3d.view_selected(ctx)
    #         bpy.ops.view3d.camera_to_view_selected(ctx)

    rd.image_settings.file_format = 'PNG'
    rd.image_settings.quality = 100

    importedObject.location = (0, 0, 0)
    importedObject.rotation_euler = (0, 0, 0)

    #MakeCameraLookAt(camobj, importedObject)
    #pose, K = GetPose(camobj, importedObject)
    #WriteMatrix('pose.txt', pose, 4)
    #WriteMatrix('K.txt', K, 3)
    #rd.filepath = 'out.png'
    #bpy.ops.render.render(write_still=True)

    return camobj, lampObj, importedObject, rd

objects = ['ape', 'benchvise', 'bowl', 'cam', 'can',
           'cat', 'cup', 'driller', 'duck', 'eggbox',
           'glue', 'holepuncher', 'iron', 'lamp', 'phone']
# objects = ['ape']
linemod_path = '/data/LINEMOD/models/'
out_path = './render_out/'
K = None

for obj in objects:
    modelFile = linemod_path + obj + '.ply'
    outputDir = out_path + obj

    # set render environment
    camobj, lampObj, importedObject, rd = SetRenderingEnv(modelFile)
    # render
    Render(outputDir, camobj, lampObj, importedObject, rd)

    # get intrinsics matrix K
    tk = get_calibration_matrix_K_from_blender(camobj, rd)
    if K == None:
        K = tk
        WriteMatrix(out_path + 'K.txt', K, 3)
    else:
        assert(K == tk)


