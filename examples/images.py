import sys
sys.path.append("../")
from pipeline import Pipeline
pipeline = Pipeline()
pipeline.chdir("images")

def processor(input, output, params):
    open(output.filename, 'w').close()

def summarizer(input, output, params):
    open(output.filename, 'w').close()

RawImage = pipeline.file("{camera}_{scene}.jpg", name="RawImage")
ProcessedImage = pipeline.file("{camera}_{scene}_processed.jpg", name="ProcessedImage")
Summary = pipeline.file("summary.txt", name="Summary")

pipeline.step(processor, RawImage, ProcessedImage)
pipeline.step(summarizer, ProcessedImage.all(), Summary)
success = pipeline.run()
