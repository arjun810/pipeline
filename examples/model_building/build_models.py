import sys
import argparse
import glob
import os
sys.path.append("../../")
from ziang import Pipeline, Task, BinaryTask, Master

class DetectChessboard(Task):
    input = {'image': 'filename'}
    output = {'board': 'filename'}

    def run(self):
        from scipy.misc import imread
        import pycb
        board_size = self.params['board_size']
        img = imread(self.input['image'])
        corners, chessboards = pycb.extract_chessboards(img, use_corner_thresholding=False)
        pycb.save_chessboard(self.output['board'], corners, chessboards, [board_size])


class UndistortCorners(Task):
    input = {'calibration': 'filename',
             'corners': 'filename'}
    output = {'undistorted_corners': 'filename'}

    def run(self):
        import h5py
        import cycloud
        import pycb
        import numpy as np

        reference_camera = self.params["reference_camera"]
        board_size = self.params["board_size"]

        calibration = h5py.File(self.input['calibration'], "r")
        K = np.array(calibration[reference_camera + "_rgb_K"])
        d = np.array(calibration[reference_camera + "_rgb_d"])
        calibration.close()

        corners = pycb.read_chessboard(self.input['corners'], board_size)

        undistorted = cycloud.undistortPoints(corners, K, d)

        with open(self.output['undistorted_corners'], 'w') as f:
            for i in range(undistorted.shape[0]):
                f.write("%f, %f\n"  % (undistorted[i, 0], undistorted[i, 1]))

class PCLVoxelGrid(BinaryTask):

    input = {'cloud': 'filename'}
    output = {'cloud': 'filename'}

    executable = "pcl_voxel_grid"

    def args(self):
        args = "{0} {1} -leaf {2}"
        return args.format(self.input['cloud'],
                           self.output['cloud'],
                           self.params['leaf_size'])

def add_chessboard_detection_tasks(pipeline, set, object):
    light_setting = pipeline.get_global('light_setting')
    reference_camera = pipeline.get_global('reference_camera')

    base_dir = os.path.join(pipeline.root_dir, set, light_setting, object)

    image_filenames = glob.glob(os.path.join(base_dir, reference_camera + "_*.jpg"))

    for image_filename in image_filenames:
        output_filename = os.path.splitext(image_filename)[0] + "_corners.txt"
        input = {"image": image_filename}
        output = {"board": output_filename}
        pipeline.add_task(DetectChessboard, input, output)

def add_corner_undistortion_tasks(pipeline, set, object, calibration_filename):
    light_setting = pipeline.get_global('light_setting')
    reference_camera = pipeline.get_global('reference_camera')

    base_dir = os.path.join(pipeline.root_dir, set, light_setting, object)

    corners_filenames = glob.glob(os.path.join(base_dir, reference_camera + "*_corners.txt"))

    for corners_filename in corners_filenames:
        output_filename = os.path.splitext(corners_filename)[0] + "_undistorted.txt"
        input = {"calibration": calibration_filename,
                 "corners": corners_filename}
        output = {"undistorted_corners": output_filename}
        pipeline.add_task(UndistortCorners, input, output)

def get_object_list(root, set, light_setting):
    dir = os.path.join(root, set, light_setting)
    objects = glob.glob(dir + "/*")
    objects = [o.split("/")[-1] for o in objects]
    return sorted(objects)

def main(sets, object):

    pipeline = Pipeline()
    pipeline.set_root("~/test_bigbird")

    board_size = {'y': 9, 'x': 8}
    reference_camera = "NP5"
    light_setting = "light_100_100_20_20"

    highres_cameras = ["N1", "N2", "N3", "N4", "N5"]
    rgbd_cameras = ["NP1", "NP2", "NP3", "NP4", "NP5"]
    rgb_cameras = highres_cameras + rgbd_cameras

    pipeline.add_globals({
        "board_size": board_size,
        "reference_camera": reference_camera,
        "light_setting": light_setting,
        "highres_cameras": highres_cameras,
        "rgbd_cameras": rgbd_cameras,
        "rgb_cameras": rgb_cameras
    })

    base_dir = pipeline.root_dir

    for set in sets:

        calibration_filename = os.path.join(base_dir,
                                            set,
                                            "calibration/calibration.h5")

        if object == "all":
            objects = get_object_list(base_dir, set, light_setting)
        else:
            objects = [object]

        for object in objects:
            #add_chessboard_detection_tasks(pipeline, set, object)
            add_corner_undistortion_tasks(pipeline, set, object, calibration_filename)
            #add_discontinuity_filtering_tasks(pipeline, set, object)
            #add_cloud_creation_tasks(pipeline, set, object)
            #add_pose_estimation_tasks(pipeline, set, object)
            #add_cloud_segmentation_tasks(pipeline, set, object)
            #add_cloud_merging_tasks(pipeline, set, object)
            #add_object_cloud_tasks(pipeline, set, object)
            #add_cloud_smoothing_tasks(pipeline, set, object)
            #add_voxelizing_tasks(pipeline, set, object)
            #add_poisson_tasks(pipeline, set, object)
            #add_tsdf_fusion_tasks(pipeline, set, object)
            #add_tsdf_mesh_tasks(pipeline, set, object)
            #add_normal_correctino_tasks(pipeline, set, object)
            #add_mesh_texturing_tasks(pipeline, set, object)
            #add_mask_generation_tasks(pipeline, set, object)

    success = pipeline.run_local_tornado()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default=None)
    parser.add_argument("--object", default="all")
    args = parser.parse_args()

    sets = []
    if args.set is None:
        sets = ["set_08_19_12_17PM"]
    elif args.set == "all":
        sets = ["set_08_19_12_17PM"]
    else:
        sets = [args.set]

    if args.object != "all" and len(sets) > 1:
        raise Exception("Can't specify a single object for multiple sets")

    print "Running on sets: "
    print sets
    print "Running on objects: "
    print args.object

    main(sets, args.object)

#import collections
#import os
#import glob
#import multiprocessing
#import sys
#import itertools
#import re
#from time import sleep
#import argparse
#import subprocess
#
#import h5py
#import numpy as np
#import cv2
#from scipy.misc import imread
#
#import pycb
#import cycloud
#
#sys.path.append("..")
#from turntable.circle_fitter import plane_equalizer
#from turntable.fitter import TurntableFitter
#
#def which(program):
#    import os
#    def is_exe(fpath):
#        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)
#
#    fpath, fname = os.path.split(program)
#    if fpath:
#        if is_exe(program):
#            return program
#    else:
#        for path in os.environ["PATH"].split(os.pathsep):
#            path = path.strip('"')
#            exe_file = os.path.join(path, program)
#            if is_exe(exe_file):
#                return exe_file
#    return None
#
#if which("pcd2ply"):
#    pcd2ply = "pcd2ply"
#elif which("pcl_pcd2ply"):
#    pcd2ply = "pcl_pcd2ply"
#else:
#    raise Exception("Need pcd2ply")
#
#base_dir = "/mnt/data/bigbird/"
#
#rgb_cameras = ["N1", "N2", "N3", "N4", "N5"]
##rgbd_cameras = ["N1", "N2", "N3", "N4", "N5"]
#rgbd_cameras = ["NP1", "NP2", "NP3", "NP4", "NP5"]
#all_cameras = rgb_cameras + rgbd_cameras
#reference_camera = "NP5"
#board_size = {'y': 9, 'x': 8}
#
## **UNITS**
#turntable_square_size = 0.018923 # == 0.745 inches
#depth_map_scale_factor = .0001 # 100um to meters
#
#def board_detector((filename, board_size)):
#    output_filename = os.path.splitext(filename)[0] + "_corners.txt"
#
#    if os.path.exists(output_filename):
#        #print "Using existing corners file: %s" % output_filename
#        return
#
#    try:
#        img = imread(filename)
#    except:
#        print "Couldn't load file %s. Exception thrown" % filename
#        return
#    if len(img.shape) == 0:
#        print "Couldn't load file %s. img had no values" % filename
#        return
#    try:
#        corners, chessboards = pycb.extract_chessboards(img, use_corner_thresholding=False)
#    except Exception as e:
#        print "Exception on file", filename
#        raise e
#
#    pycb.save_chessboard(output_filename, corners, chessboards, [board_size])
#
#def detect_reference_chessboards(dir, reference_camera, board_size, num_processes=12):
#
#    print "Detecting boards.."
#    p = multiprocessing.Pool(num_processes)
#    image_filenames = glob.glob(os.path.join(dir, reference_camera + "_*.jpg"))
#    results = p.map_async(board_detector, zip(image_filenames, itertools.repeat(board_size)))
#    last_num_detected = 0
#    while not results.ready():
#        num_detected = len(glob.glob(os.path.join(dir, reference_camera + "*corn*.txt")))
#        if num_detected != last_num_detected:
#            print "Detected %d/%d" % (num_detected, len(image_filenames))
#            last_num_detected = num_detected
#        sleep(0.5)
#    print "Done detecting boards."
#    p.terminate()
#
#def corner_undistorter((filename, K, d, board_size)):
#    output_filename = os.path.splitext(filename)[0] + "_undistorted.txt"
#
#    if os.path.exists(output_filename):
#        #print "Using existing corners file: %s" % output_filename
#        return
#
#    try:
#        corners = pycb.read_chessboard(filename, board_size)
#    except:
#        print "Couldn't load file %s. Exception thrown" % filename
#        return
#
#    undistorted = cycloud.undistortPoints(corners, K, d)
#
#    with open(output_filename, 'w') as f:
#        for i in range(undistorted.shape[0]):
#            f.write("%f, %f\n"  % (undistorted[i, 0], undistorted[i, 1]))
#
#def undistort_corners(dir, reference_camera, board_size, calibration_filename, num_processes=12):
#
#    calibration = h5py.File(calibration_filename, "r")
#
#    K = np.array(calibration[reference_camera + "_rgb_K"])
#    d = np.array(calibration[reference_camera + "_rgb_d"])
#
#    calibration.close()
#
#    print "Undistorting corners.."
#    p = multiprocessing.Pool(num_processes)
#    corners_filenames = glob.glob(os.path.join(dir, reference_camera + "*_corners.txt"))
#    results = p.map_async(corner_undistorter, zip(corners_filenames, itertools.repeat(K), itertools.repeat(d), itertools.repeat(board_size)))
#    last_num_undistorted = 0
#    while not results.ready():
#        num_undistorted = len(glob.glob(os.path.join(dir, reference_camera + "*corn*undis*.txt")))
#        if num_undistorted != last_num_undistorted:
#            print "undistorted %d/%d" % (num_undistorted, len(corners_filenames))
#            last_num_undistorted = num_undistorted
#        sleep(0.5)
#    print "Done undistorting corners."
#    p.terminate()
#
#def discontinuity_filterer((dir, camera, scene)):
#    input_file = os.path.join(dir, "NP%s_%s.h5" % (camera[-1], scene))
#    if not os.path.exists(input_file):
#        print "Can't find input file %s" % input_file
#        return
#    output_file = os.path.join(dir, "%s_%s_discontinuity_filtered.h5" % (camera, scene))
#    if os.path.exists(output_file):
#        return
#    command = "python ../util/depth_discontinuity.py --in_loc %s --out_loc %s" % (input_file, output_file)
#    os.system(command)
#
#def filter_discontinuities(dir, cameras, scenes, num_processes=12):
#    tasks = []
#    for camera in cameras:
#        for scene in scenes:
#            tasks.append((dir, camera, scene))
#    p = multiprocessing.Pool(num_processes)
#    results = p.map_async(discontinuity_filterer, tasks)
#    last_num_filtered = 0
#    while not results.ready():
#        num_filtered = len(glob.glob(os.path.join(dir, "*discont*")))
#        if num_filtered != last_num_filtered:
#            print "Filtered %d/%d" % (num_filtered, len(tasks))
#            last_num_filtered = num_filtered
#        sleep(0.5)
#    if not results.successful():
#        print results.get()
#    p.terminate()
#
#def cloud_creator((rgb_K, depth_K, rgb_D, ir_D, H_rgb_from_depth, base_filename, depth_scale, depth_map_scale_factor)):
#    if os.path.exists(base_filename + ".pcd"):
#        return
#
#    with h5py.File(base_filename + "_discontinuity_filtered.h5") as disc_file:
#        disc_filtered_map = disc_file["depth"].value
#    image = imread(base_filename + ".jpg")
#    unregistered = disc_filtered_map * depth_map_scale_factor * depth_scale
#    window = 21
#    sigma_depth = 0.00635 # in meters (0.25 inches) **UNITS**
#    sigma_pixels = 10.5
#    filtered = cycloud.bilateral_filter(unregistered, window, sigma_depth, sigma_pixels)
#    registered = cycloud.registerDepthMap(filtered, image, depth_K, rgb_K, H_rgb_from_depth)
#
#    #unregistered_undistorted = cycloud.undistortDepthMap(unregistered, depth_K, ir_D)
#    #registered = cycloud.registerDepthMap(unregistered_undistorted, image, depth_K, rgb_K, H_rgb_from_depth, rgb_D)
#
#    cloud = cycloud.registeredDepthMapToPointCloud(registered, image, rgb_K, organized=False)
#    cycloud.writePCD(cloud, base_filename + ".pcd")
#
#def create_clouds(dir, reference_camera, cameras, scenes, calibration_filename, depth_map_scale_factor, num_processes=8):
#    print "Creating clouds.."
#
#    if not os.path.exists(calibration_filename):
#        raise IOError("{0} does not exist.".format(calibration_filename))
#    calibration = h5py.File(calibration_filename, "r")
#
#    tasks = []
#
#    for camera in cameras:
#        rgb_K = np.array(calibration[camera + "_rgb_K"])
#        rgb_D = np.array(calibration["NP" + camera[-1] + "_rgb_d"])
#        depth_K = np.array(calibration["NP" + camera[-1] + "_depth_K"])
#        #depth_K[1,2] -= 12
#        ir_D = np.array(calibration["NP" + camera[-1] + "_ir_d"])
#        depth_scale = np.array(calibration["NP" + camera[-1] + "_ir_depth_scale"])
#        H_rgb_from_ref = calibration["H_" + camera + "_from_" + reference_camera][:]
#        H_ir_from_ref = calibration["H_NP" + camera[-1] + "_ir_from_" + reference_camera][:]
#        H_rgb_from_depth = np.dot(H_rgb_from_ref, np.linalg.inv(H_ir_from_ref))
#        for scene in scenes:
#            base = os.path.join(dir, camera + "_" + scene)
#            tasks.append((rgb_K, depth_K, rgb_D, ir_D, H_rgb_from_depth, base, depth_scale, depth_map_scale_factor))
#
#    p = multiprocessing.Pool(num_processes)
#    results = p.map_async(cloud_creator, tasks)
#    #results = map(cloud_creator, tasks)
#    last_num_created = 0
#    while not results.ready():
#        #num_created = len(glob.glob(os.path.join(dir, "*organized.pcd")))
#        num_created = len(glob.glob(os.path.join(dir, "*.pcd")))
#        if num_created != last_num_created:
#            print "Created %d/%d" % (num_created, len(tasks))
#            last_num_created = num_created
#        sleep(0.5)
#    if not results.successful():
#        print results.get()
#
#    calibration.close()
#    p.terminate()
#
#def cloud_merger((dir, output_dir, cameras, scene, reference_camera, camera_transforms)):
#    output_filename = os.path.join(output_dir, "scene_" + scene + ".pcd")
#    if os.path.exists(output_filename):
#        return
#    clouds = {}
#    total_points = 0
#    for camera in cameras:
#        cloud_filename = os.path.join(dir, "segmented_clouds", camera + "_" + scene + ".pcd")
#        if os.path.exists(cloud_filename):
#            clouds[camera] = cycloud.readPCD(cloud_filename)
#            total_points += clouds[camera].shape[1]
#        else:
#            clouds[camera] = None
#    merged_cloud = np.empty((1, total_points, 6))
#    offset = 0
#    for camera in cameras:
#        if clouds[camera] is None:
#            continue
#        num_points = clouds[camera].shape[1]
#        cycloud.transformCloud(clouds[camera], camera_transforms[camera], inplace=True)
#        merged_cloud[:, offset:offset+num_points, :] = clouds[camera]
#        offset += num_points
#    cycloud.writePCD(merged_cloud, output_filename)
#
#def merge_clouds_single_view(dir, cameras, scenes, reference_camera, calibration_filename, output_dir="merged_scenes", num_processes=12):
#    print "Merging clouds"
#    output_dir = os.path.join(dir, output_dir)
#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#    calibration = h5py.File(calibration_filename, "r")
#    tasks = []
#    camera_transforms = {}
#
#    for camera in cameras:
#        name = "H_{0}_from_{1}".format(camera, reference_camera)
#        camera_transforms[camera] = np.linalg.inv(np.array(calibration[name]))
#    for scene in scenes:
#        tasks.append((dir, output_dir, cameras, scene, reference_camera, camera_transforms))
#
#    p = multiprocessing.Pool(num_processes)
#    results = p.map_async(cloud_merger, tasks)
#    last_num_merged = 0
#    while not results.ready():
#        num_merged = len(glob.glob(os.path.join(output_dir, "*.pcd")))
#        if num_merged != last_num_merged:
#            print "Merged %d/%d" % (num_merged, len(tasks))
#            last_num_merged = num_merged
#        sleep(0.5)
#    if not results.successful():
#        print results.get()
#    calibration.close()
#    p.terminate()
#
#def cloud_segmenter((camera_name, scene, output_dir, segmenter_params)):
#    """
#    segment the particular cam's view
#    """
#    basename = "{0}_{1}.pcd".format(camera_name, scene)
#    output_filename = os.path.join(output_dir, basename)
#    if os.path.exists(output_filename):
#        return
#    (dir, calibration_filename, board_params, reference_camera, pose_detector) = segmenter_params
#    input_filename = os.path.join(dir, basename)
#    outlier_radius = 0.005 # **UNITS**
#    min_neighbors = 5
#    reference_pose_basename = "{0}_{1}_pose.h5".format(reference_camera, scene)
#    reference_pose_path = os.path.join(dir, "poses", pose_detector, reference_pose_basename)
#    turntable_path = os.path.join(dir, "poses", pose_detector, "turntable.h5")
#    command = "../build/segment_and_filter "
#    command += "--in={0} ".format(input_filename)
#    command += "--out={0} ".format(output_filename)
#    command += "--calibration_path={0} ".format(calibration_filename)
#    command += "--camera_name={0} ".format(camera_name)
#    command += "--reference_camera_name={0} ".format(reference_camera)
#    command += "--reference_pose_path={0} ".format(reference_pose_path)
#    command += "--turntable_path={0} ".format(turntable_path)
#    command += "--radius={0} ".format(outlier_radius)
#    command += "--min_neighbors={0} ".format(min_neighbors)
#    command += "--board_x={0} ".format(board_params['x'])
#    command += "--board_y={0} ".format(board_params['y'])
#    command += "--square_size={0} ".format(board_params['square_size'])
#    os.system(command)
#
#def segment_clouds(dir, cams, scenes, output_dir='segmented_clouds', num_processes=12):
#    print "Segmenting clouds"
#    output_dir = os.path.join(dir, output_dir)
#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#    board_params = {'x': board_size['x'],
#                    'y': board_size['y'],
#                    'square_size': turntable_square_size}
#    tasks = []
#    for cam in cams:
#        for scene in scenes:
#            tasks.append((cam, scene, output_dir, (dir, calibration_filename, board_params, reference_camera, "optimized")))
#
#    p = multiprocessing.Pool(num_processes)
#    results = p.map_async(cloud_segmenter, tasks)
#    #results = map(cloud_segmenter, tasks)
#    last_num_segmented = 0
#    while not results.ready():
#        num_segmented = len(glob.glob(os.path.join(output_dir, "*.pcd")))
#        if num_segmented != last_num_segmented:
#            print "Segmented %d/%d" % (num_segmented, len(tasks))
#            last_num_segmented = num_segmented
#        sleep(0.5)
#    if not results.successful():
#        raise Exception(str(results.get()))
#    p.terminate()
#
#def pose_estimator_solvepnp((dir, output_dir, scene, reference_camera, K, d, board_size, square_size)):
#    output_filename = os.path.join(output_dir, "{0}_{1}_pose.h5".format(reference_camera, scene))
#    if os.path.exists(output_filename):
#        return
#    points_3d = pycb.get_3d_chessboard_points(board_size['x'], board_size['y'], square_size)
#    input_filename = os.path.join(dir, "{0}_{1}_corners.txt".format(reference_camera, scene))
#    points_2d = pycb.read_chessboard(input_filename, board_size)
#
#    #result, R, t = cv2.solvePnP(points_3d, points_2d, K, d)
#    result, R, t = cv2.solvePnP(points_3d, points_2d, K, np.zeros(5))
#
#    H = np.eye(4)
#    cv2.Rodrigues(R, H[:3, :3])
#    H[:3, 3] = t[:, 0]
#    H = np.linalg.inv(H)
#    file = h5py.File(output_filename)
#    file["H_table_from_reference_camera"] = H
#    file.close()
#
#def estimate_reference_poses_solvepnp(base_dir, scenes, reference_camera, board_size, square_size, calibration_filename, output_dir="poses/solvepnp", num_processes=12):
#    print "estimating poses..."
#    output_dir = os.path.join(base_dir, output_dir)
#    calibration = h5py.File(calibration_filename, "r")
#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#
#    K = np.array(calibration[reference_camera + "_rgb_K"])
#    d = np.array(calibration[reference_camera + "_rgb_d"])
#    calibration.close()
#
#    tasks = []
#    for scene in scenes:
#        tasks.append((dir, output_dir, scene, reference_camera, K, d, board_size, square_size))
#
#    p = multiprocessing.Pool(num_processes)
#    results = p.map_async(pose_estimator_solvepnp, tasks)
#    #results = map(pose_estimator_solvepnp, tasks)
#
#    last_num_estimated = 0
#    while not results.ready():
#        num_estimated = len(glob.glob(os.path.join(output_dir, "*.h5")))
#        if num_estimated != last_num_estimated:
#            print "Estimated pose on %d/%d" % (num_estimated, len(tasks))
#            last_num_estimated = num_estimated
#        sleep(0.5)
#    if not results.successful():
#        print results.get()
#    p.terminate()
#
#def estimate_reference_poses_circlefit(base_dir, scenes, reference_camera, output_dir="poses/circlefit"):
#    print "estimating poses..."
#
#    output_dir = os.path.join(base_dir, output_dir)
#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#
#    points_dir = os.path.join(base_dir, "poses/solvepnp")
#    poses = []
#    for scene in scenes:
#        f = h5py.File(os.path.join(points_dir, "%s_%s_pose.h5" % (reference_camera, scene)))
#        pose = np.linalg.inv(f["H_table_from_reference_camera"].value)
#        poses.append(pose)
#        f.close()
#
#    better_poses = plane_equalizer(poses)
#    for i, scene in enumerate(scenes):
#        better_pose = better_poses[i]
#        output_filename = os.path.join(output_dir, "{0}_{1}_pose.h5".format(reference_camera, scene))
#        if os.path.exists(output_filename):
#            continue
#        file = h5py.File(output_filename)
#        file["H_table_from_reference_camera"] = np.linalg.inv(better_pose)
#        file.close()
#
#    #import pickle
#    #pickle.dump(better_poses, open(output_dir + "/testpickle", 'w'))
#
#def estimate_reference_poses_optimized(base_dir, dataset, lighting, object, reference_camera, square_size, board_size, dir_name="optimized"):
#    fitter = TurntableFitter(base_dir,
#                             dataset,
#                             lighting,
#                             object,
#                             reference_camera,
#                             square_size,
#                             board_size)
#    fitter.fit()
#    fitter.save_poses(dir_name)
#
#def create_object_cloud(dir, cams, reference_camera, pose_sources, scenes, output_dir="object_clouds"):
#    print "Creating object cloud..."
#    output_dir = os.path.join(dir, output_dir)
#    # TODO
#    cams = [None]# + cams
#    calibration = h5py.File(calibration_filename, "r")
#    def output_filename(pose_source, cam):
#        if cam is None:
#            return os.path.join(output_dir, "%s.pcd" % pose_source)
#        else:
#            return os.path.join(output_dir, "%s_%s.pcd" % (cam, pose_source))
#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#    for pose_source in pose_sources:
#        cam_merged_clouds = dict((cam, np.empty((1, 10000000, 6))) for cam in cams)
#        num_points = collections.defaultdict(int)
#        n_scenes_failed = collections.defaultdict(int)
#        for scene in scenes:
#            for cam in cams:
#                if os.path.exists(output_filename(pose_source, cam)):
#                    continue
#                try:
#                    if cam is None:
#                        pcd_file = os.path.join(dir, "merged_scenes", "scene_%s.pcd" % (scene))
#                        cloud = cycloud.readPCD(pcd_file)
#                    else:
#                        pcd_file = os.path.join(dir, "segmented_clouds", "{0}_{1}.pcd".format(cam, scene))
#                        cloud = cycloud.readPCD(pcd_file)
#                        if cam != reference_camera:
#                            name = "H_{0}_from_{1}".format(cam, reference_camera)
#                            T = np.linalg.inv(np.array(calibration[name]))
#                            cycloud.transformCloud(cloud, T, inplace=True)
#
#                    pose_filename = os.path.join(dir, "poses", pose_source, "%s_%s_pose.h5" % (reference_camera, scene))
#                    pose_file = h5py.File(pose_filename, 'r')
#                    H_table_from_camera = np.array(pose_file["H_table_from_reference_camera"])
#                    pose_file.close()
#
#                except IOError:
#                    n_scenes_failed[cam] += 1
#                    continue
#
#                merged_cloud = cam_merged_clouds[cam]
#                cycloud.transformCloud(cloud, H_table_from_camera, inplace=True)
#                points_in_cloud = cloud.shape[1]
#                if num_points[cam] + points_in_cloud > merged_cloud.shape[1]:
#                    new_merged_cloud = np.empty((1, merged_cloud.shape[1]*2, 6))
#                    new_merged_cloud[:,:merged_cloud.shape[1],:] = merged_cloud
#                    merged_cloud = new_merged_cloud
#                merged_cloud[:, num_points[cam]:num_points[cam]+points_in_cloud,:] = cloud
#                num_points[cam] += points_in_cloud
#        for cam in cams:
#            if os.path.exists(output_filename(pose_source, cam)):
#                continue
#            if n_scenes_failed[cam]:
#                print n_scenes_failed[cam], 'scenes failed for pose_source', pose_source, 'cam', cam
#            print 'WRITING', num_points[cam], 'POINTS FOR', cam, 'IN', pose_source
#            merged_cloud = cam_merged_clouds[cam][:,:num_points[cam],:]
#            cycloud.writePCD(merged_cloud, output_filename(pose_source, cam))
#    calibration.close()
#
#def cloud_voxelizer((input_file, output_file, no_downsample)):
#    output_ply = os.path.splitext(input_file)[0] + ".ply"
#    if not os.path.exists(output_ply):
#        command = "%s %s %s >/dev/null" % (pcd2ply, input_file, output_ply)
#        os.system(command)
#    if os.path.exists(output_file):
#        return
#    if not no_downsample:
#        #**UNITS**
#        command = "pcl_voxel_grid %s %s -leaf 0.000635,0.000635,0.000635" % (input_file, output_file)
#    else:
#        command = "cp %s %s" % (input_file, output_file)
#    os.system(command)
#    command = "%s %s %s >/dev/null" % (pcd2ply, output_file, os.path.splitext(output_file)[0] + ".ply")
#    os.system(command)
#
#def voxelize_clouds(dir, cams, pose_sources, output_dir="object_clouds", num_processes=5):
#    output_dir = os.path.join(dir, output_dir)
#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#    tasks = []
#    for pose_source in pose_sources:
#        for cam in [None]:#TODO + cams:
#            if cam is None:
#                input_file = os.path.join(output_dir, pose_source + ".pcd")
#                output_file = os.path.join(output_dir, pose_source + "_voxelized.pcd")
#                tasks.append((input_file, output_file, False))
#                input_file = os.path.join(output_dir, pose_source + "_smoothed.pcd")
#                output_file = os.path.join(output_dir, pose_source + "_smoothed_voxelized.pcd")
#                tasks.append((input_file, output_file, False))
#            else:
#                input_file = os.path.join(output_dir, '{0}_{1}.pcd'.format(cam, pose_source))
#                output_file = os.path.join(output_dir, '{0}_{1}_voxelized.pcd'.format(cam, pose_source))
#                tasks.append((input_file, output_file, False))
#
#    p = multiprocessing.Pool(num_processes)
#    results = p.map_async(cloud_voxelizer, tasks)
#
#    last_num_voxelized = 0
#    while not results.ready():
#        num_voxelized = len(glob.glob(os.path.join(output_dir, "*.h5")))
#        if num_voxelized != last_num_voxelized:
#            print "Voxelized %d/%d" % (num_voxelized, len(tasks))
#            last_num_voxelized = num_voxelized
#        sleep(0.5)
#    if not results.successful():
#        print results.get()
#    p.terminate()
#
#def smooth_object_clouds(dir, pose_sources, num_processes=1):
#    tasks = []
#    for pose_source in pose_sources:
#        input_file = os.path.join(dir, "object_clouds", pose_source + ".pcd")
#        output_file = os.path.join(dir, "object_clouds", pose_source + "_smoothed.pcd")
#        tasks.append((input_file, output_file))
#
#    results = map(smoother, tasks)
#
#def smoother((input_file, output_file)):
#    if os.path.exists(output_file):
#        return
#    #command = "../../utils/build/np_smoother --in_loc %s --out_loc %s" % (input_file, output_file)
#    command = "../build/filter_merged --in %s --out %s" % (input_file, output_file)
#    print command
#    os.system(command)
#
#def create_poisson_meshes(dir, pose_sources, output_dir="meshes", num_processes=2):
#    output_dir = os.path.join(dir, output_dir)
#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#    tasks = []
#    for pose_source in pose_sources:
#        input_file = os.path.join(dir, "object_clouds", pose_source + "_smoothed_voxelized.ply")
#        output_file = os.path.join(output_dir, pose_source + "_voxelized_mesh.ply")
#        tasks.append((input_file, output_file))
#        #input_file = os.path.join(dir, "object_clouds", pose_source + "_smoothed.ply")
#        #output_file = os.path.join(output_dir, pose_source + "_mesh.ply")
#        #tasks.append((input_file, output_file))
#
#    p = multiprocessing.Pool(num_processes)
#    results = p.map_async(poisson_meshmaker, tasks)
#
#    last_num_meshed = 0
#    while not results.ready():
#        num_meshed = len(glob.glob(os.path.join(output_dir, "*.h5")))
#        if num_meshed != last_num_meshed:
#            print "Meshed %d/%d" % (num_meshed, len(tasks))
#            last_num_meshed = num_meshed
#        sleep(0.5)
#    if not results.successful():
#        print results.get()
#    p.terminate()
#
#def poisson_meshmaker((input_file, output_file)):
#    if os.path.exists(output_file):
#        return
#    command = "meshlabserver -i %s -o %s -s normals_and_poisson.mlx" % (input_file, output_file)
#    print command
#    os.system(command)
#
#
#def create_tsdf(data_dir, calib_day, light, object_type, pose_types,
#                output_dir="tsdf"):
#    print "Creating TSDFs..."
#    output_dir = os.path.join(
#        data_dir, calib_day, light, object_type, output_dir)
#    if not os.path.exists(output_dir):
#      os.makedirs(output_dir)
#
#    for pose_type in pose_types:
#      # First, construct the TSDF
#      output_filename = os.path.join(
#          output_dir, "{0}_marching_cubes_mesh.tsdf".format(pose_type))
#      if os.path.exists(output_filename):
#        continue
#      binary = "../tsdf/build/tsdf_runner"
#      command = ("%s --data_directory=%s --calibration_day=%s "
#                 "--lighting_settings=%s --object=%s --pose_type=%s "
#                 "--tsdf_type tsdf_calibration_without_align "
#                 "--camera -1 --out_loc=%s --verbose" % (
#                     binary, data_dir, calib_day, light, object_type,
#                     pose_type, output_filename))
#      print command
#      os.system(command)
#
#
#def create_tsdf_meshes(data_dir, calib_day, light, object_type,
#                       pose_types, output_dir="meshes"):
#    print "Creating TSDF meshes..."
#    object_dir = os.path.join(data_dir, calib_day, light, object_type)
#    if not os.path.exists(output_dir):
#      os.makedirs(output_dir)
#
#    for pose_type in pose_types:
#      # First, construct the TSDF
#      input_filename = os.path.join(
#          object_dir, "tsdf",
#          "{0}_marching_cubes_mesh.tsdf".format(pose_type))
#      output_filename = os.path.join(
#          object_dir, output_dir, "{0}_marching_cubes_mesh.ply".format(
#              pose_type))
#      if os.path.exists(output_filename):
#        continue
#      binary = "../tsdf/build/tsdf_to_mesh"
#      command = ("../tsdf/build/tsdf_to_mesh --in_loc {0} "
#                 "--out_loc {1}".format(input_filename, output_filename))
#      print command
#      os.system(command)
#
#def texture_meshes(data_dir, calib_day, light, object_type,
#                   pose_types, output_folder="textured_meshes"):
#    print "Texturing meshes..."
#    base_dir = os.path.join(data_dir, calib_day)
#    object_dir = os.path.join(base_dir, light, object_type)
#
#    output_dir = os.path.join(object_dir, output_folder) + "/"
#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#
#    for pose_type in pose_types:
#        meshname = "{0}_marching_cubes_mesh.ply".format(pose_type)
#        outfile = os.path.join(output_dir, "{0}_tsdf_textured_mesh.ply".format(pose_type))
#        if os.path.exists(outfile):
#            continue
#        command = ("../build/texture_mesh --base_dir %s --light %s --object %s --posetype %s \
#                --meshname %s --outfile %s --side_camera %s --upsample 0 --top_camera %s"\
#                % (base_dir, light, object_type, pose_type, meshname, outfile, "N2", "N4"))
#        print command
#        os.system(command)
#
#        outname = "{0}_tsdf_texture_mapped_mesh".format(pose_type)
#        command = ("../build/texture_map_mesh --base_dir %s --light %s --object %s --posetype %s \
#                --meshname %s --outdir %s --outname %s --side_camera %s --upsample 0 --top_camera %s" \
#                % (base_dir, light, object_type, pose_type, meshname, output_dir, outname, "N2", "N4"))
#        print command
#        os.system(command)
#
#        meshname = "{0}_voxelized_mesh.ply".format(pose_type)
#        outfile = os.path.join(output_dir, "{0}_poisson_textured_mesh.ply".format(pose_type))
#        if os.path.exists(outfile):
#            continue
#        command = ("../build/texture_mesh --base_dir %s --light %s --object %s --posetype %s \
#                --meshname %s --outfile %s --side_camera %s --upsample 0 --top_camera %s" \
#                % (base_dir, light, object_type, pose_type, meshname, outfile, "N2", "N4"))
#        print command
#        os.system(command)
#
#        outname = "{0}_poisson_texture_mapped_mesh".format(pose_type)
#        command = ("../build/texture_map_mesh --base_dir %s --light %s --object %s --posetype %s \
#                --meshname %s --outdir %s --outname %s --side_camera %s --upsample 0 --top_camera %s" \
#                % (base_dir, light, object_type, pose_type, meshname, output_dir, outname, "N2", "N4"))
#        print command
#        os.system(command)
#
#def generate_masks(base_dir, reference_camera, calibration, calibration_filename,
#        light_setting, object, cameras, scenes, pose_type, mesh_type):
#    object_dir = os.path.join(base_dir, calibration, light_setting, object)
#    output_dir = os.path.join(object_dir, "masks")
#    if not os.path.exists(output_dir):
#        os.makedirs(output_dir)
#    tasks = []
#    if mesh_type == "poisson":
#        mesh_file = os.path.join(object_dir, "meshes", "{0}_voxelized_mesh.ply".format(pose_type))
#    else:
#        mesh_file = os.path.join(object_dir, "meshes", "{0}_marching_cubes_mesh.ply".format(pose_type))
#    if not os.path.exists(mesh_file):
#        print "Cannot generate masks, mesh doesn't exist: {0}".format(mesh_file)
#        return
#
#    for camera in cameras:
#        for scene in scenes:
#            image_file = os.path.join(object_dir, "{0}_{1}.jpg".format(camera, scene))
#            output_file = os.path.join(output_dir, "{0}_{1}_mask.pbm".format(camera, scene))
#            pose_file = os.path.join(object_dir, "poses", pose_type, "{0}_{1}_pose.h5".format(reference_camera, scene))
#            tasks.append((reference_camera, camera, scene, calibration_filename, pose_file, mesh_file, image_file, output_file))
#
#    p = multiprocessing.Pool(10)
#    results = p.map_async(maskmaker, tasks)
#
#    last_num_masked = 0
#    while not results.ready():
#        num_masked = len(glob.glob(os.path.join(output_dir, "*.pbm")))
#        if num_masked != last_num_masked:
#            print "masked %d/%d" % (num_masked, len(tasks))
#            last_num_masked = num_masked
#        sleep(0.5)
#    if not results.successful():
#        print results.get()
#    p.terminate()
#
#def maskmaker((reference_camera, camera, scene, calibration_filename, pose_file, mesh_file, image_file, output_file)):
#    if os.path.exists(output_file):
#        return
#    command = "../build/segmentation_masks --reference_camera {0} --camera {1} --scene {2}"
#    command += " --calibration {3} --pose {4} --mesh {5} --image {6} --outfile {7}"
#    command = command.format(reference_camera, camera, scene, calibration_filename, pose_file, mesh_file, image_file, output_file)
#    os.system(command)
#
#def fix_normals(data_dir, calib_day, light, object_type, pose_types):
#    base_dir = os.path.join(data_dir, calib_day)
#    object_dir = os.path.join(base_dir, light, object_type)
#    commands = []
#    for pose_type in pose_types:
#        mesh_file = os.path.join(object_dir, "meshes", "{0}_voxelized_mesh.ply".format(pose_type))
#        command = ["../build/correct_mesh_normals", "--in_path", mesh_file]
#        commands.append(command)
#
#        mesh_file = os.path.join(object_dir, "meshes", "{0}_marching_cubes_mesh.ply".format(pose_type))
#        command = ["../build/correct_mesh_normals", "--in_path", mesh_file]
#        commands.append(command)
#
#    for command in commands:
#        output = subprocess.Popen(command, stdout=subprocess.PIPE)
#        out, err = output.communicate()
#        if out.rstrip() == "1":
#            inv_command = "meshlabserver -i %s -o %s -s invert_normals.mlx" % (command[2], command[2])
#            print(inv_command)
#            os.system(inv_command)
#
#
#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--set", default=None)
#    parser.add_argument("--object", default="detergent")
#    args = parser.parse_args()
#
#    calibrations = []
#    if args.set is None:
#        #calibrations = ["set_05_02_05_03PM"]
#        #calibrations = ["set_05_05_02_30PM"]
#        #calibrations = ["set_05_08_04_00PM"]
#        calibrations = ["set_05_09_03_00PM"]
#        #calibrations = ["set_05_10_04_30PM"]
#        calibrations = ["set_08_19_12_17PM"]
#    elif args.set == "all":
#        calibrations.append("set_05_02_05_03PM")
#        calibrations.append("set_05_05_02_30PM")
#        calibrations.append("set_05_08_04_00PM")
#        calibrations.append("set_05_09_03_00PM")
#        #calibrations.append("set_05_10_04_30PM")
#    else:
#        calibrations = [args.set]
#
#    if args.object != "all" and len(calibrations) > 1:
#        raise Exception("Can't specify a single object for multiple calibrations")
#
#    print "Running on calibrations: "
#    print calibrations
#    print "Running on objects: "
#    print args.object
#
#    light_setting = "light_100_100_20_20"
#
#    for calibration in calibrations:
#        calibration_filename = os.path.join(base_dir, calibration, "calibration/calibration.h5")
#        if args.object == "all":
#            objects = glob.glob(os.path.join(base_dir, calibration, light_setting) + "/*")
#            objects = [o.split("/")[-1] for o in objects]
#            objects = sorted(objects)
#        else:
#            objects = [args.object]
#        for object in objects:
#            dir = os.path.join(base_dir, calibration, light_setting, object)
#            #if os.path.exists(os.path.join(dir, "textured_meshes", "optimized_tsdf_textured_mesh.ply")):
#            #    print "skipping %s" % object
#            #    continue
#            if object == "melissa_doug_play-time_produce_farm_fresh_fruit_unopened_box":
#                continue
#            if object == "sterilite_bin_12qt_bottom":
#                continue
#
#            all_scenes = [str(x) for x in range(0,360,3)]
#
#            #os.system("rm %s/*undistorted.txt" % dir)
#            #os.system("rm %s/*discont*" % dir)
#            #os.system("rm %s/*.pcd" % dir)
#            #os.system("rm -r %s/*" % os.path.join(dir, "poses"))
#            #os.system("rm %s/*" % os.path.join(dir, "segmented_clouds"))
#            #os.system("rm %s/*" % os.path.join(dir, "merged_scenes"))
#            #os.system("rm %s/*" % os.path.join(dir, "object_clouds"))
#            os.system("rm %s/*" % os.path.join(dir, "meshes"))
#            #os.system("rm %s/*" % os.path.join(dir, "tsdf"))
#            os.system("rm %s/*" % os.path.join(dir, "textured_meshes"))
#            os.system("rm %s/*" % os.path.join(dir, "masks"))
#
#            #detect_reference_chessboards(dir, reference_camera, board_size)
#            #undistort_corners(dir, reference_camera, board_size, calibration_filename)
#            #filter_discontinuities(dir, rgbd_cameras, all_scenes)
#            #create_clouds(dir, reference_camera, rgbd_cameras, all_scenes, calibration_filename, depth_map_scale_factor)
#
#            scenes = [str(x) for x in range(0,360,3)]
#
#            #estimate_reference_poses_solvepnp(dir, scenes, reference_camera, board_size, turntable_square_size, calibration_filename)
#            #estimate_reference_poses_circlefit(dir, scenes, reference_camera)
#            #estimate_reference_poses_optimized(base_dir,
#            #                                calibration,
#            #                                light_setting,
#            #                                object,
#            #                                reference_camera,
#            #                                turntable_square_size,
#            #                                board_size)
#
#            #segment_clouds(dir, rgbd_cameras, scenes)
#
#            #merge_clouds_single_view(dir, rgbd_cameras, scenes, reference_camera, calibration_filename)
#
#            #posetypes = ["solvepnp", "circlefit", "optimized"]
#            #posetypes = ["optimized", "solvepnp"]
#            posetypes = ["optimized"]
#            #create_object_cloud(dir, rgbd_cameras, reference_camera, posetypes, scenes)
#            #smooth_object_clouds(dir, posetypes)
#            #voxelize_clouds(dir, rgbd_cameras, posetypes)
#
#            create_poisson_meshes(dir, posetypes)
#
#            #create_tsdf(base_dir, calibration, light_setting, object, posetypes)
#            create_tsdf_meshes(base_dir, calibration, light_setting, object,
#                            posetypes)
#            fix_normals(base_dir, calibration, light_setting, object, posetypes)
#            texture_meshes(base_dir, calibration, light_setting, object, posetypes)
#            generate_masks(base_dir, reference_camera, calibration, calibration_filename,
#                        light_setting, object, all_cameras, all_scenes, "optimized", "tsdf")
#            os.system("rm %s/*.pcd" % dir)
#
