import sys
import os
import glob

from unittest import TestCase
from nose.tools import eq_, raises
from nose.plugins.attrib import attr

from nose_parameterized import parameterized

from pipeline import Pipeline, FilePattern

fixtures_abspath = os.path.join(os.path.dirname(__file__), "fixtures")

def no_op(inputs, outputs):
    pass

def iter_len(iterator):
    return len(list(iterator))

def iter_len_eq(iterator, value):
    eq_(iter_len(iterator), value)

class FilePatternTest(TestCase):

    def test_get_pattern_root(self):
        pattern = FilePattern("/a/b/c", "d/{var1}/e/{var2}/g.txt")
        pattern_root = pattern.pattern_root()
        eq_(pattern_root, "/a/b/c/d")

        pattern = FilePattern("/a/b/c", "{var1}/e/{var2}/g.txt")
        pattern_root = pattern.pattern_root()
        eq_(pattern_root, "/a/b/c")

    @raises(Exception)
    def test_absolute_pattern_invalid(self):
        pattern = FilePattern("/a/b/c", "/d/{var1}/e/{var2}/g.txt")

    def test_discover(self):
        path = os.path.join(fixtures_abspath, "test1")

        pattern = FilePattern(path, "{camera}_{scene}.jpg")
        expected_files = glob.glob(os.path.join(path, "*.jpg"))
        pattern.discover()
        self.assertItemsEqual(pattern.filenames, expected_files)
        self.assertItemsEqual(pattern.variables, ['camera', 'scene'])

        pattern = FilePattern(path, "{camera}_{scene}.h5")
        expected_files = glob.glob(os.path.join(path, "*.h5"))
        pattern.discover()
        self.assertItemsEqual(pattern.filenames, expected_files)
        self.assertItemsEqual(pattern.variables, ['camera', 'scene'])
        for instance in pattern.instances:
            assert('camera' in instance)
            assert('scene' in instance)

        pattern = FilePattern(path, "NP1_{scene}.h5")
        expected_files = glob.glob(os.path.join(path, "NP1*.h5"))
        pattern.discover()
        self.assertItemsEqual(pattern.filenames, expected_files)
        self.assertItemsEqual(pattern.variables, ['scene'])
        for i, expected_file in enumerate(expected_files):
            for instance in pattern.instances:
                if instance['filename'] == expected_file:
                    # Note: this is very test-specific. It relies on the scene
                    # being the fourth-from-last character, which is true for a
                    # single-digit scene with a file with an extension with two
                    # characters.
                    eq_(instance['scene'], expected_file[-4])
                    assert('camera' not in instance)

    @parameterized.expand([
        ("{camera}_{scene}.h5", ["camera", "scene"]),
        ("{camera}/{scene}.h5", ["camera", "scene"]),
        ("{camera}_scene.h5", ["camera"]),
        ("camera/{scene}.h5", ["scene"]),
    ])
    def test_variables(self, path, expected_variables):
        root = os.path.join(fixtures_abspath, "test1")
        pattern = FilePattern(root, path)
        self.assertItemsEqual(pattern.variables, expected_variables)

    @parameterized.expand([
        ("{camera}_{scene}.h5", {"camera": "N1", "scene": 1}, "N1_1.h5"),
        ("{camera}/{scene}.h5", {"camera": "N2", "scene": "4"}, "N2/4.h5"),
        ("{camera}_scene.h5",   {"camera": "N1"}, "N1_scene.h5"),
        ("camera/{scene}.h5",   {"scene": 0}, "camera/0.h5"),
    ])
    def test_format(self, path, values, expected_basename):
        root = os.path.join(fixtures_abspath, "test1")
        pattern = FilePattern(root, path)
        expected = os.path.join(root, expected_basename)
        self.assertEqual(pattern.format(values), expected)

class PipelineChdirTest(TestCase):

    def test_chdir(self):
        pipeline = Pipeline()

        path = "fixtures/test1"
        pipeline.chdir(path)
        expected_path = os.path.join(os.path.dirname(__file__), os.pardir, path)
        expected_path = os.path.abspath(expected_path)
        eq_(expected_path, pipeline.current_dir)

        path = "./fixtures/test1"
        pipeline.chdir(path)
        expected_path = os.path.abspath(path)
        eq_(expected_path, pipeline.current_dir)

        path = "/fixtures/test1"
        pipeline.chdir(path)
        expected_path = path
        eq_(expected_path, pipeline.current_dir)

class PipelineResourceTest(TestCase):

    def setUp(self):
        self.pipeline = Pipeline()

    @raises(Exception)
    def test_resource_without_chdir(self):
        pipeline = Pipeline()
        RawImage = pipeline.file("{camera}_{scene}.jpg")

    def test_input_resource(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg")

        iter_len_eq(RawImage, 6)

        iter_len_eq(RawImage.all(), 1)
        iter_len_eq(RawImage.each(), 6)

        iter_len_eq(RawImage.where(camera="N1"), 2)
        iter_len_eq(RawImage.where(scene=0), 3)
        iter_len_eq(RawImage.where(scene=0, camera="N1"), 1)

        iter_len_eq(RawImage.where(camera="N1").each(), 2)
        iter_len_eq(RawImage.where(scene=0).each(), 3)
        iter_len_eq(RawImage.where(scene=0, camera="N1").each(), 1)

        iter_len_eq(RawImage.where(camera="N1").all(), 1)
        iter_len_eq(RawImage.where(scene=0).all(), 1)
        iter_len_eq(RawImage.where(scene=0, camera="N1").all(), 1)

        raw_image = RawImage.where(scene=0).first()
        eq_(raw_image["scene"], "0")

    @raises(ValueError)
    def test_invalid_condition(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg")
        RawImage.where(badkey="N1")

    def test_nonexistent_resource_query(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg")
        iter_len_eq(RawImage.where(scene=5), 0)

    @raises(ValueError)
    def test_nonexistent_resource_query_first(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg")
        RawImage.where(scene=5).first()

    @parameterized.expand([
        (0,),
        (1,),
    ])
    def test_build_resource_same_vars(self, scene_number):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg")
        ProcessedImage = self.pipeline.file("{camera}_{scene}_processed.jpg")

        raw_image = RawImage.where(scene=scene_number).first()
        processed_image = ProcessedImage.build(raw_image)
        eq_(raw_image["camera"], processed_image["camera"])
        eq_(raw_image["scene"], processed_image["scene"])
        expected_basename = "{0}_{1}_processed.jpg".format(raw_image["camera"],
                                                           raw_image["scene"])
        expected_filename = os.path.join(path, expected_basename)
        eq_(expected_filename, processed_image["filename"])

    @raises(ValueError)
    def test_build_resource_subset_vars(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg")
        ProcessedImage = self.pipeline.file("{camera}_processed.jpg")

        raw_image = RawImage.where(scene=0).first()
        processed_image = ProcessedImage.build(raw_image)

    @raises(ValueError)
    def test_build_resource_extra_vars(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg")
        ProcessedImage = self.pipeline.file("{camera}_{scene}_{extra}.jpg")

        raw_image = RawImage.where(scene=0).first()
        processed_image = ProcessedImage.build(raw_image)

    def test_build_resource_each(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg")
        ProcessedImage = self.pipeline.file("{camera}_{scene}_processed.jpg")

        raw_images = RawImage.all()

        for raw_image in RawImage.each():
            processed_image = ProcessedImage.build(raw_image)
            eq_(raw_image["scene"], processed_image["scene"])
            eq_(raw_image["camera"], processed_image["camera"])

        raw_image = RawImage.where(camera="N1",scene=0).first()
        processed_image = ProcessedImage.build(raw_image)
        eq_(raw_image['scene'], processed_image['scene'])
        eq_(raw_image['camera'], processed_image['camera'])

    def test_build_resource_all(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg")
        Summary = self.pipeline.file("summary.txt")

        summary = Summary.build(next(RawImage.all()))
        expected = os.path.join(path, 'summary.txt')
        eq_(summary['filename'], expected)

    def test_build_resource_all_conditions_1(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg")
        Summary = self.pipeline.file("summary.txt")

        summary = Summary.build(next(RawImage.where(camera="N1").all()))
        expected = os.path.join(path, 'summary.txt')
        eq_(summary['filename'], expected)

    @parameterized.expand([
        (0,),
        (1,),
    ])
    def test_build_resource_all_conditions_2(self, scene_number):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg")
        Summary = self.pipeline.file("{scene}_summary.txt")

        raw_images = next(RawImage.where(scene=scene_number).all())
        summary = Summary.build(raw_images)

        expected_basename = '{0}_summary.txt'.format(scene_number)
        expected = os.path.join(path, expected_basename)
        eq_(summary['filename'], expected)
        eq_(summary['scene'], str(scene_number))
        eq_(summary.get('camera', None), None)

    @raises(ValueError)
    def test_build_resource_all_conditions_check_subset_1(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg")
        Summary = self.pipeline.file("{bad}_summary.txt")
        raw_images = next(RawImage.where(camera="N1").all())
        summary = Summary.build(raw_images)

    @raises(ValueError)
    def test_build_resource_all_conditions_check_subset_2(self):
        """
        Ensure all inputs have all required keys for the output resource
        """
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg")
        Summary = self.pipeline.file("{bad}_summary.txt")

        raw_images = next(RawImage.where(camera="N1").all())
        raw_images[0]["bad"] = 4
        summary = Summary.build(raw_images)

    @raises(ValueError)
    def test_build_resource_all_retained_variables_same_values(self):
        """
        This test ensures that all inputs have the same values for any variable
        used by the output resource.
        """
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg")
        Summary = self.pipeline.file("{camera}_summary.txt")

        raw_images = next(RawImage.all())
        summary = Summary.build(raw_images)

    def test_build_resource_grouped(self):
        # TODO this should work
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg")
        Summary = self.pipeline.file("{camera}_summary.txt")

        for group in RawImage.group_by("camera"):
            summary = Summary.build(group)
            expected_basename = '{0}_summary.txt'.format(group[0]['camera'])
            expected = os.path.join(path, expected_basename)
            eq_(summary['camera'], group[0]['camera'])
            eq_(summary['filename'], expected)
            eq_(summary.get('scene', None), None)

    def test_build_resource_grouped_is_subset(self):
        # TODO this should raise error
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg")
        Summary = self.pipeline.file("{camera}_{bad}_summary.txt")

        group = RawImage.group_by("camera").first()
        summary = Summary.build(group)

    @attr('skip')
    def test_invalid_output_resource_spec(self):
        fail
        pass
        #TODO



class PipelineStepTest(TestCase):

    def setUp(self):
        self.pipeline = Pipeline()
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)

    def test_list_io(self):
        self.pipeline.step(no_op, ["a.txt", "b.txt"], ["c.txt", "d.txt"])
        eq_(len(self.pipeline.queue), 2)

        self.pipeline.step(no_op, ["a.txt"], ["c.txt"])
        eq_(len(self.pipeline.queue), 3)

    @raises(ValueError)
    def test_invalid_list_io_1(self):
        self.pipeline.step(no_op, ["a.txt"], ["c.txt", "d.txt"])

    @raises(ValueError)
    def test_invalid_list_io_2(self):
        self.pipeline.step(no_op, ["a.txt", "b.txt"], ["c.txt"])

    def test_each_io(self):
        RawImage = self.pipeline.file("{camera}_{scene}.jpg")
        ProcessedImage = self.pipeline.file("{camera}_{scene}_processed.jpg")
        self.pipeline.step(no_op, RawImage.each(), ProcessedImage)

        eq_(len(self.pipeline.queue), iter_len(RawImage))

class FullPipelineTest(TestCase):

    def setUp(self):
        self.pipeline = Pipeline()

    def test_input_resource(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg")
