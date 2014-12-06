import sys
sys.path.append("../../")
from ziang import Pipeline, Task

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

success = pipeline.run()
#pipeline.compute_job_graph()
#print pipeline.job_graph.graph.nodes()
#print pipeline.resource_job_graph.graph.nodes()
