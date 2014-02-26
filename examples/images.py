import sys
sys.path.append("../")
from pipeline import Pipeline
pipeline = Pipeline()
pipeline.chdir("images")

def processor(input, output, params):
    open(output.filename, 'w').close()

def summarizer(input, output, params):
    open(output.filename, 'w').close()

RawImage = pipeline.file("{camera}_{scene}.jpg")
ProcessedImage = pipeline.file("{camera}_{scene}_processed.jpg")
Summary = pipeline.file("summary.txt")

pipeline.step(processor, RawImage, ProcessedImage)
pipeline.step(summarizer, ProcessedImage.all(), Summary)
success = pipeline.run()
