import sys
sys.path.append("../../")
from ziang import Pipeline, Task, BinaryTask

class ImageProcessor(Task):

    input = {'image': 'filename'}
    output = {'image': 'filename'}

    def run(self):
        open(self.output['image'], 'w').close()

class ImageSummarizer(Task):

    input = {'images': 'filename_list'}
    output = {'summary': 'filename'}

    def run(self):
        open(self.output['summary'], 'w').close()

class PCLVoxelGrid(BinaryTask):

    input = {'cloud': 'filename'}
    output = {'cloud': 'filename'}

    executable = "pcl_voxel_grid"

    def args(self):
        args = "{0} {1} -leaf {2}"
        return args.format(self.input['cloud'],
                           self.output['cloud'],
                           self.params['leaf_size'])

pipeline = Pipeline()
pipeline.set_root("data")

cameras = ['N1', 'NP1', 'NP2']
scenes = [0, 1]

for camera in cameras:
    for scene in scenes:
        input = {'image': "{0}_{1}.jpg".format(camera, scene)}
        output = {'image': "{0}_{1}_processed.jpg".format(camera, scene)}
        pipeline.add_task(ImageProcessor, input, output)

input = {}
input['images'] = []
for camera in cameras:
    for scene in scenes:
        input['images'].append("{0}_{1}_processed.jpg".format(camera, scene))
output = {'summary': "summary.txt"}
pipeline.add_task(ImageSummarizer, input, output)

pipeline.add_task(PCLVoxelGrid, {"cloud": "scene_0.pcd"}, {"cloud": "scene_0_voxelized.pcd"}, leaf_size=.01)

success = pipeline.run_local_tornado()
print "Returned"
#pipeline.compute_job_graph()
#print pipeline.job_graph.graph.nodes()
#print pipeline.resource_job_graph.graph.nodes()
