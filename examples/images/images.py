import sys
sys.path.append("../../")
from ziang import Pipeline, Task
pipeline = Pipeline()
pipeline.chdir("data")

class ImageProcessor(Task):

    input = {'image': 'dummy'}
    output = {'image': 'dummy'}

    def run(self):
        output_filename = self.output['image'].filename
        open(output_filename, 'w').close()

class ImageSummarizer(Task):

    input = {'images': ['dummy']}
    output = {'summary': 'dummy'}

    def run(self):
        output_filename = self.output['summary'].filename
        open(output_filename, 'w').close()

RawImage = pipeline.file("{camera}_{scene:\d+}.jpg")
ProcessedImage = pipeline.file("{camera}_{scene}_processed.jpg")
Summary = pipeline.file("summary.txt")

pipeline.step(ImageProcessor, RawImage, ProcessedImage)
pipeline.step(ImageSummarizer, ProcessedImage.all(), Summary)
success = pipeline.run()
