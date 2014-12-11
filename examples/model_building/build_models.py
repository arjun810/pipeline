import sys
import argparse
import glob
import os
sys.path.append("../../")
from ziang import Pipeline, Task, BinaryTask, Master

PERCEPTION_DIR =  "/home/arjun/Documents/v/perception"

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

class FilterDepthDiscontinuities(BinaryTask):
    input = {'depth_map': 'filename'}
    output = {'filtered_depth_map': 'filename'}

    executable = os.path.join(PERCEPTION_DIR, "model_building", "util", "depth_discontinuity.py")

    def args(self):
        args = "--in_loc {0} --out_loc {1}"
        return args.format(self.input['depth_map'],
                           self.output['filtered_depth_map'])

class CreateCloud(Task):
    input = {'calibration': 'filename',
             'filtered_depth_map': 'filename',
             'image': 'filename'}
    output = {'cloud': 'filename'}

    def run(self):
        import h5py
        import cycloud
        import numpy as np
        from scipy.misc import imread

        depth_map_scale_factor = self.params["depth_map_scale_factor"]
        reference_camera = self.params["reference_camera"]
        camera = self.params["camera"]

        calibration = h5py.File(self.input['calibration'], "r")
        K = np.array(calibration[reference_camera + "_rgb_K"])
        d = np.array(calibration[reference_camera + "_rgb_d"])
        rgb_K = np.array(calibration[camera + "_rgb_K"])
        rgb_D = np.array(calibration["NP" + camera[-1] + "_rgb_d"])
        depth_K = np.array(calibration["NP" + camera[-1] + "_depth_K"])
        ir_D = np.array(calibration["NP" + camera[-1] + "_ir_d"])
        depth_scale = np.array(calibration["NP" + camera[-1] + "_ir_depth_scale"])
        H_rgb_from_ref = calibration["H_" + camera + "_from_" + reference_camera][:]
        H_ir_from_ref = calibration["H_NP" + camera[-1] + "_ir_from_" + reference_camera][:]
        H_rgb_from_depth = np.dot(H_rgb_from_ref, np.linalg.inv(H_ir_from_ref))
        calibration.close()

        with h5py.File(self.input["filtered_depth_map"]) as disc_file:
            disc_filtered_map = disc_file["depth"].value

        unregistered = disc_filtered_map * depth_map_scale_factor * depth_scale

        window = 21
        sigma_depth = 0.00635 # in meters (0.25 inches) **UNITS**
        sigma_pixels = 10.5
        filtered = cycloud.bilateral_filter(unregistered, window, sigma_depth, sigma_pixels)

        image = imread(self.input['image'])
        registered = cycloud.registerDepthMap(filtered, image, depth_K, rgb_K, H_rgb_from_depth)

        cloud = cycloud.registeredDepthMapToPointCloud(registered, image, rgb_K, organized=False)
        cycloud.writePCD(cloud, self.output['cloud'])

class EstimatePoses(BinaryTask):
    input = {'corner_files': 'filename_list'}
    output = {'pose_files': 'filename_list',
              'turntable': 'filename'}

    executable = os.path.join(PERCEPTION_DIR, "model_building", "turntable", "fitter.py")

    def args(self):
        args = "--base_path {0} --dataset {1} --lighting {2} --object {3} --camera {4} --board_x {5} --board_y {6} --square_size {7}"
        return args.format(self.params['base_path'],
                           self.params['dataset'],
                           self.params['light_setting'],
                           self.params['object'],
                           self.params['reference_camera'],
                           self.params['board_size']['x'],
                           self.params['board_size']['y'],
                           self.params['turntable_square_size'])

class SegmentCloud(BinaryTask):

    executable = os.path.join(PERCEPTION_DIR, "model_building", "build", "segment_and_filter")

    input = {'cloud': 'filename',
             'calibration': 'filename',
             'reference_pose': 'filename',
             'turntable': 'filename',
             }
    output = {'cloud': 'filename'}

    def args(self):
        args = "--in {0} --out {1} -calibration_path {2} \
                --camera_name {3} --reference_camera_name {4} \
                --reference_pose_path {5} --turntable_path {6} \
                --radius {7} --min_neighbors {8} \
                --board_x {9} --board_y {10} --square_size {11}"

        return args.format(self.input['cloud'],
                           self.output['cloud'],
                           self.input['calibration'],
                           self.params['camera_name'],
                           self.params['reference_camera'],
                           self.input['reference_pose'],
                           self.input['turntable'],
                           self.params['radius'],
                           self.params['min_neighbors'],
                           self.params['board_size']['x'],
                           self.params['board_size']['y'],
                           self.params['turntable_square_size'])

class MergeClouds(Task):
    input = {'clouds': 'filename_list',
             'calibration': 'filename'}
    output = {'cloud': 'filename'}

    def run(self):
        import h5py
        import cycloud
        import numpy as np

        reference_camera = self.params['reference_camera']

        clouds = {}
        total_points = 0
        camera_transforms = {}
        calibration = h5py.File(self.input['calibration'], "r")
        for camera, cloud_filename in zip(self.params['cameras'], self.input['clouds']):
            # The file might not exist if the segmented cloud had 0 points.
            if not os.path.exists(cloud_filename):
                clouds[camera] = None
                continue
            clouds[camera] = cycloud.readPCD(cloud_filename)
            total_points += clouds[camera].shape[1]

            name = "H_{0}_from_{1}".format(camera, reference_camera)
            camera_transforms[camera] = np.linalg.inv(np.array(calibration[name]))
        calibration.close()

        merged_cloud = np.empty((1, total_points, 6))
        offset = 0
        for camera in self.params['cameras']:
            if clouds[camera] is None:
                continue
            num_points = clouds[camera].shape[1]
            cycloud.transformCloud(clouds[camera], camera_transforms[camera], inplace=True)
            merged_cloud[:, offset:offset+num_points, :] = clouds[camera]
            offset += num_points

        cycloud.writePCD(merged_cloud, self.output['cloud'])

class CreateObjectCloud(Task):
    input = {'clouds': 'filename_list',
             'poses': 'filename_list',
             'calibration': 'filename'}
    output = {'cloud': 'filename'}

    def run(self):
        import h5py
        import cycloud
        import numpy as np

        clouds = {}
        num_points = 0
        camera_transforms = []
        merged_cloud = np.empty((1, 10000000, 6))
        #calibration = h5py.File(self.input['calibration'], "r")
        #calibration.close()
        for cloud_filename, pose_filename in zip(self.input['clouds'], self.input['poses']):
            cloud = cycloud.readPCD(cloud_filename)

            pose_file = h5py.File(pose_filename, 'r')
            H_table_from_camera = np.array(pose_file["H_table_from_reference_camera"])
            pose_file.close()

            cycloud.transformCloud(cloud, H_table_from_camera, inplace=True)

            points_in_cloud = cloud.shape[1]
            if num_points + points_in_cloud > merged_cloud.shape[1]:
                new_merged_cloud = np.empty((1, merged_cloud.shape[1]*2, 6))
                new_merged_cloud[:,:merged_cloud.shape[1],:] = merged_cloud
                merged_cloud = new_merged_cloud

            merged_cloud[:, num_points:num_points+points_in_cloud,:] = cloud
            num_points += points_in_cloud

        merged_cloud = merged_cloud[:,:num_points,:]
        cycloud.writePCD(merged_cloud, self.output['cloud'])

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

def add_discontinuity_filtering_tasks(pipeline, set, object):
    light_setting = pipeline.get_global('light_setting')
    base_dir = os.path.join(pipeline.root_dir, set, light_setting, object)

    for rgbd_camera in pipeline.get_global('rgbd_cameras'):
        for scene in pipeline.get_global('scenes'):
            input_filename = os.path.join(base_dir, "{0}_{1}.h5".format(rgbd_camera, scene))
            output_filename = os.path.join(base_dir, "{0}_{1}_discontinuity_filtered.h5".format(rgbd_camera, scene))
            input = {"depth_map": input_filename}
            output = {"filtered_depth_map": output_filename}
            pipeline.add_task(FilterDepthDiscontinuities, input, output)

def add_cloud_creation_tasks(pipeline, set, object, calibration_filename):
    light_setting = pipeline.get_global('light_setting')
    base_dir = os.path.join(pipeline.root_dir, set, light_setting, object)

    for rgbd_camera in pipeline.get_global('rgbd_cameras'):
        for scene in pipeline.get_global('scenes'):
            depth_map_filename = os.path.join(base_dir, "{0}_{1}_discontinuity_filtered.h5".format(rgbd_camera, scene))
            image_filename = os.path.join(base_dir, "{0}_{1}.jpg".format(rgbd_camera, scene))
            cloud_filename = os.path.join(base_dir, "{0}_{1}.pcd".format(rgbd_camera, scene))
            input = {'calibration': calibration_filename,
                    'filtered_depth_map': depth_map_filename,
                    'image': image_filename}
            output = {'cloud': cloud_filename}
            pipeline.add_task(CreateCloud, input, output, camera=rgbd_camera)

def add_pose_estimation_tasks(pipeline, set, object, calibration_filename):
    reference_camera = pipeline.get_global('reference_camera')
    light_setting = pipeline.get_global('light_setting')
    base_dir = os.path.join(pipeline.root_dir, set, light_setting, object)
    pose_dir = os.path.join(base_dir, 'poses', 'optimized')

    corner_files = []
    pose_files = []
    for scene in pipeline.get_global('scenes'):
        corner_file = os.path.join(base_dir, "{0}_{1}_undistorted.txt".format(reference_camera, scene))
        pose_file = os.path.join(pose_dir, "{0}_{1}_pose.h5".format(reference_camera, scene))
        corner_files.append(corner_file)
        pose_files.append(pose_file)

    turntable_filename = os.path.join(pose_dir, 'turntable.h5')
    input = {'corner_files': corner_files}
    output = {'pose_files': pose_files, 'turntable': turntable_filename}
    pipeline.add_task(EstimatePoses, input, output,
                      base_path=pipeline.root_dir,
                      dataset=set,
                      object=object)

def add_cloud_segmentation_tasks(pipeline, set, object, calibration_filename):

    light_setting = pipeline.get_global('light_setting')
    reference_camera = pipeline.get_global('reference_camera')
    base_dir = os.path.join(pipeline.root_dir, set, light_setting, object)
    pose_dir = os.path.join(base_dir, 'poses', 'optimized')

    outlier_radius = 0.005 # **UNITS**
    min_neighbors = 5

    for rgbd_camera in pipeline.get_global('rgbd_cameras'):
        for scene in pipeline.get_global('scenes'):
            input_filename = os.path.join(base_dir, "{0}_{1}.pcd".format(rgbd_camera, scene))
            reference_pose = os.path.join(pose_dir, "{0}_{1}_pose.h5".format(reference_camera, scene))
            turntable = os.path.join(pose_dir, "turntable.h5")
            output_filename = os.path.join(base_dir, "segmented_clouds", "{0}_{1}.pcd".format(rgbd_camera, scene))
            input = {'cloud': input_filename,
                    'calibration': calibration_filename,
                    'reference_pose': reference_pose,
                    'turntable': turntable,
                    }
            output = {'cloud': output_filename}
            pipeline.add_task(SegmentCloud, input, output,
                              camera_name=rgbd_camera,
                              radius=outlier_radius,
                              min_neighbors=min_neighbors)

def add_cloud_merging_tasks(pipeline, set, object, calibration_filename):

    reference_camera = pipeline.get_global('reference_camera')
    light_setting = pipeline.get_global('light_setting')
    base_dir = os.path.join(pipeline.root_dir, set, light_setting, object)
    pose_dir = os.path.join(base_dir, 'poses', 'optimized')

    for scene in pipeline.get_global('scenes'):
        input_clouds = []
        for rgbd_camera in pipeline.get_global('rgbd_cameras'):
            input_filename = os.path.join(base_dir, "segmented_clouds", "{0}_{1}.pcd".format(rgbd_camera, scene))
            input_clouds.append(input_filename)

        input = {'clouds': input_clouds,
                'calibration': calibration_filename,
                }
        output_filename = os.path.join(base_dir, "merged_scenes", "scene_{0}.pcd".format(scene))
        output = {'cloud': output_filename}

        pipeline.add_task(MergeClouds, input, output, cameras=pipeline.get_global('rgbd_cameras'))

def add_object_cloud_tasks(pipeline, set, object, calibration_filename):
    reference_camera = pipeline.get_global('reference_camera')
    light_setting = pipeline.get_global('light_setting')
    base_dir = os.path.join(pipeline.root_dir, set, light_setting, object)
    pose_dir = os.path.join(base_dir, 'poses', 'optimized')

    cloud_filenames = []
    pose_filenames = []
    for scene in pipeline.get_global('scenes'):
        input_filename = os.path.join(base_dir, "merged_scenes", "scene_{0}.pcd".format(scene))
        pose_filename = os.path.join(pose_dir, "{0}_{1}_pose.h5".format(reference_camera, scene))
        cloud_filenames.append(input_filename)
        pose_filenames.append(pose_filename)

    output_filename = os.path.join(base_dir, "object_clouds", "optimized.pcd")
    input = {'clouds': cloud_filenames,
             'poses': pose_filenames,
             'calibration': calibration_filename}
    output = {'cloud': output_filename}
    pipeline.add_task(CreateObjectCloud, input, output)

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
    turntable_square_size = 0.018923 # == 0.745 inches
    depth_map_scale_factor = .0001 # 100um to meters

    highres_cameras = ["N1", "N2", "N3", "N4", "N5"]
    rgbd_cameras = ["NP1", "NP2", "NP3", "NP4", "NP5"]
    rgb_cameras = highres_cameras + rgbd_cameras
    scenes = range(0,360,3)

    pipeline.add_globals({
        "board_size": board_size,
        "reference_camera": reference_camera,
        "light_setting": light_setting,
        "highres_cameras": highres_cameras,
        "rgbd_cameras": rgbd_cameras,
        "rgb_cameras": rgb_cameras,
        "scenes": scenes,
        "turntable_square_size": turntable_square_size,
        "depth_map_scale_factor": depth_map_scale_factor,
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
            add_chessboard_detection_tasks(pipeline, set, object)
            add_corner_undistortion_tasks(pipeline, set, object, calibration_filename)
            add_discontinuity_filtering_tasks(pipeline, set, object)
            add_cloud_creation_tasks(pipeline, set, object, calibration_filename)
            add_pose_estimation_tasks(pipeline, set, object, calibration_filename)
            add_cloud_segmentation_tasks(pipeline, set, object, calibration_filename)
            add_cloud_merging_tasks(pipeline, set, object, calibration_filename)
            add_object_cloud_tasks(pipeline, set, object, calibration_filename)
            #add_cloud_smoothing_tasks(pipeline, set, object)
            #add_voxelizing_tasks(pipeline, set, object)
            #add_poisson_tasks(pipeline, set, object)
            #add_tsdf_fusion_tasks(pipeline, set, object)
            #add_tsdf_mesh_tasks(pipeline, set, object)
            #add_normal_correction_tasks(pipeline, set, object)
            #add_mesh_texturing_tasks(pipeline, set, object)
            #add_mask_generation_tasks(pipeline, set, object)

    return pipeline
    #success = pipeline.run_local_tornado()
    #success = pipeline.run()

def test(pipeline, n_computers):
    from ziang.scheduling import *

    objects = [
        "sharpie_marker",
        "small_black_spring_clamp",
        "soft_scrub_2lb_4oz",
        "spam_12oz",
        "sponge_with_textured_cover",
        "stainless_steel_fork_red_handle",
        "stainless_steel_knife_red_handle",
        "stainless_steel_spatula",
        "stainless_steel_spoon_red_handle",
        "stanley_13oz_hammer",
        "stanley_flathead_screwdriver",
        "stanley_philips_screwdriver",
        "starkist_chunk_light_tuna",
        "sterilite_bin_12qt_bottom",
        "sterilite_bin_12qt_cap",
    ]

    dg=pipeline.resource_job_graph
    tg = generate_task_graph(dg)
    assert any(resource.data["final"] for resource in tg.get_resources())

    for job in tg.get_jobs():
        job.data["job_dur"] = 1.0

    for resource in tg.get_resources():
        resource.data["send_dur"] = 3.0

    idealjob2loc = {}
    for job in tg.get_jobs():
        output = job.outputs[0]
        if isinstance(output, list):
            output = output[0]
        for i, object in enumerate(objects):
            if object in output:
                idealjob2loc[job.name] = i % n_computers
    cost_ideal = compute_cost_with_assignment(tg, idealjob2loc, n_computers)
    print "TIME FROM HEURISTIC", cost_ideal

    assert any(resource.data["final"] for resource in tg.get_resources())
    # plan_with_ilp(tg)
    # plan_with_djikstra(tg,2)
    cost_naive,job2loc_naive = compute_cost_naive(tg,n_computers,return_job2loc=True)
    print "TIME FROM NAIVE PLANNER", cost_naive
    assert cost_naive == compute_cost_with_assignment(tg, job2loc_naive, n_computers)

    print "RUNNING HILL-CLIMBING PLANNER INITIALIZED FROM IDEAL"
    cost_hc,job2loc_hc = plan_with_hill_climb(tg,n_computers,initialize=idealjob2loc)
    print "TIME FROM HILL-CLIMBING PLANNER INITIALIZED WITH IDEAL",cost_hc
    assert cost_hc == compute_cost_with_assignment(tg, job2loc_hc, n_computers)

    print "RUNNING HILL-CLIMBING PLANNER INITIALIZED FROM NAIVE PLANNER"
    cost_hc,job2loc_hc = plan_with_hill_climb(tg,n_computers,initialize=job2loc_naive)
    print "TIME FROM HILL-CLIMBING PLANNER INITIALIZED WITH NAIVE",cost_hc
    assert cost_hc == compute_cost_with_assignment(tg, job2loc_hc, n_computers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default=None)
    parser.add_argument("--object", default="all")
    parser.add_argument("--n_computers", default=10)
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

    pipeline = main(sets, args.object)
    test(pipeline, int(args.n_computers))
