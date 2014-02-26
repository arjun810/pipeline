import sys
import os
import shutil
import glob

from unittest import TestCase
from nose.tools import eq_, raises
from nose.plugins.attrib import attr

from nose_parameterized import parameterized

from pipeline import Pipeline, FilePattern, DependencyGraph

fixtures_abspath = os.path.join(os.path.dirname(__file__), "fixtures")

def no_op(input, output, params):
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
        RawImage = pipeline.file("{camera}_{scene}.jpg", name="RawImage")

    def test_resource_id(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")

        image1 = RawImage.where(camera="N1", scene=0).first()
        image2 = RawImage.where(camera="N1", scene=1).first()
        self.assertNotEqual(image1.id, image2.id)

        image1_built = RawImage(camera="N1", scene=0)
        image2_built = RawImage(camera="N1", scene=1)
        self.assertNotEqual(image1_built.id, image2_built.id)

        eq_(image1.id, image1_built.id)
        eq_(image2.id, image2_built.id)

    def test_input_resource(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")

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

        iter_len_eq(RawImage.group_by("scene"), 2)
        iter_len_eq(RawImage.group_by("camera"), 3)
        iter_len_eq(RawImage.group_by("scene").each(), 2)
        iter_len_eq(RawImage.group_by("camera").each(), 3)
        iter_len_eq(RawImage.group_by("scene").all(), 1)
        iter_len_eq(RawImage.group_by("camera").all(), 1)

        raw_image = RawImage.where(scene=0).first()
        eq_(raw_image.scene, "0")

    @raises(ValueError)
    def test_invalid_condition(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        RawImage.where(badkey="N1")

    def test_nonexistent_resource_query(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        iter_len_eq(RawImage.where(scene=5), 0)

    @raises(ValueError)
    def test_nonexistent_resource_query_first(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        RawImage.where(scene=5).first()

    @parameterized.expand([
        (0,),
        (1,),
    ])
    def test_build_resource_same_vars(self, scene_number):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        ProcessedImage = self.pipeline.file("{camera}_{scene}_processed.jpg", name="ProcessedImage")

        raw_image = RawImage.where(scene=scene_number).first()
        processed_image = ProcessedImage.build(raw_image)
        eq_(raw_image.camera, processed_image.camera)
        eq_(raw_image.scene, processed_image.scene)
        expected_basename = "{0}_{1}_processed.jpg".format(raw_image.camera,
                                                           raw_image.scene)
        expected_filename = os.path.join(path, expected_basename)
        eq_(expected_filename, processed_image.filename)

    @raises(ValueError)
    def test_build_resource_subset_vars(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        ProcessedImage = self.pipeline.file("{camera}_processed.jpg", name="ProcessedImage")

        raw_image = RawImage.where(scene=0).first()
        processed_image = ProcessedImage.build(raw_image)

    @raises(ValueError)
    def test_build_resource_extra_vars(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        ProcessedImage = self.pipeline.file("{camera}_{scene}_{extra}.jpg", name="ProcessedImage")

        raw_image = RawImage.where(scene=0).first()
        processed_image = ProcessedImage.build(raw_image)

    def test_build_resource_each(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        ProcessedImage = self.pipeline.file("{camera}_{scene}_processed.jpg", name="ProcessedImage")

        raw_images = RawImage.all()

        for raw_image in RawImage.each():
            processed_image = ProcessedImage.build(raw_image)
            eq_(raw_image.scene, processed_image.scene)
            eq_(raw_image.camera, processed_image.camera)

        raw_image = RawImage.where(camera="N1",scene=0).first()
        processed_image = ProcessedImage.build(raw_image)
        eq_(raw_image.scene, processed_image.scene)
        eq_(raw_image.camera, processed_image.camera)

    def test_build_resource_all(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        Summary = self.pipeline.file("summary.txt", name="Summary")

        summary = Summary.build(next(RawImage.all()))
        expected = os.path.join(path, 'summary.txt')
        eq_(summary.filename, expected)

    def test_build_resource_all_conditions_1(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        Summary = self.pipeline.file("summary.txt", name="Summary")

        summary = Summary.build(next(RawImage.where(camera="N1").all()))
        expected = os.path.join(path, 'summary.txt')
        eq_(summary.filename, expected)

    @parameterized.expand([
        (0,),
        (1,),
    ])
    def test_build_resource_all_conditions_2(self, scene_number):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        Summary = self.pipeline.file("{scene}_summary.txt", name="Summary")

        raw_images = next(RawImage.where(scene=scene_number).all())
        summary = Summary.build(raw_images)

        expected_basename = '{0}_summary.txt'.format(scene_number)
        expected = os.path.join(path, expected_basename)
        eq_(summary.filename, expected)
        eq_(summary.scene, str(scene_number))
        self.assertFalse(hasattr(summary, 'camera'))

    @raises(ValueError)
    def test_build_resource_all_conditions_check_subset_1(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        Summary = self.pipeline.file("{bad}_summary.txt", name="Summary")
        raw_images = next(RawImage.where(camera="N1").all())
        summary = Summary.build(raw_images)

    @raises(ValueError)
    def test_build_resource_all_conditions_check_subset_2(self):
        """
        Ensure all inputs have all required keys for the output resource
        """
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        Summary = self.pipeline.file("{bad}_summary.txt", name="Summary")

        raw_images = next(RawImage.where(camera="N1").all())
        raw_images[0].bad = 4
        summary = Summary.build(raw_images)

    @raises(ValueError)
    def test_build_resource_all_retained_variables_same_values(self):
        """
        This test ensures that all inputs have the same values for any variable
        used by the output resource.
        """
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        Summary = self.pipeline.file("{camera}_summary.txt", name="Summary")

        raw_images = next(RawImage.all())
        summary = Summary.build(raw_images)

    def test_build_resource_grouped(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        Summary = self.pipeline.file("{camera}_summary.txt", name="Summary")

        for group in RawImage.group_by("camera"):
            summary = Summary.build(group)
            expected_basename = '{0}_summary.txt'.format(group[0].camera)
            expected = os.path.join(path, expected_basename)
            eq_(summary.camera, group[0].camera)
            eq_(summary.filename, expected)
            self.assertFalse(hasattr(summary, 'scene'))

    @raises(ValueError)
    def test_build_resource_grouped_is_subset(self):
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        Summary = self.pipeline.file("{camera}_{bad}_summary.txt", name="Summary")

        group = RawImage.group_by("camera").first()
        summary = Summary.build(group)

class PipelineStepTest(TestCase):

    def setUp(self):
        self.pipeline = Pipeline()
        path = os.path.join(fixtures_abspath, "test1")
        self.pipeline.chdir(path)

    def test_each_io_1(self):
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        ProcessedImage = self.pipeline.file("{camera}_{scene}_processed.jpg", name="ProcessedImage")
        self.pipeline.step(no_op, RawImage.each(), ProcessedImage)

        eq_(len(self.pipeline.jobs), iter_len(RawImage))

    def test_each_io_2(self):
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        ProcessedImage = self.pipeline.file("{camera}_{scene}_processed.jpg", name="ProcessedImage")
        self.pipeline.step(no_op, RawImage, ProcessedImage)

        eq_(len(self.pipeline.jobs), iter_len(RawImage))

    def test_all_io(self):
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        ProcessedImage = self.pipeline.file("processed.jpg", name="ProcessedImage")
        self.pipeline.step(no_op, RawImage.all(), ProcessedImage)

        eq_(len(self.pipeline.jobs), iter_len(RawImage.all()))

    def test_grouped_io_1(self):
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        ProcessedImage = self.pipeline.file("{camera}_processed.jpg", name="ProcessedImage")
        self.pipeline.step(no_op, RawImage.group_by("camera"), ProcessedImage)

        eq_(len(self.pipeline.jobs), iter_len(RawImage.group_by("camera")))

    def test_grouped_io_2(self):
        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        ProcessedImage = self.pipeline.file("{scene}_processed.jpg", name="ProcessedImage")
        self.pipeline.step(no_op, RawImage.group_by("scene"), ProcessedImage)

        eq_(len(self.pipeline.jobs), iter_len(RawImage.group_by("scene")))


class FullPipelineTest(TestCase):

    def setUp(self):
        self.pipeline = Pipeline()
        fixture_path = os.path.join(fixtures_abspath, "test1")
        self.test_path = os.path.join(fixtures_abspath, "test1_tmp")
        shutil.copytree(fixture_path, self.test_path)
        self.pipeline.chdir(self.test_path)

    def tearDown(self):
        shutil.rmtree(self.test_path)

    def test_single_step(self):

        def processor(input, output, params):
            open(output.filename, 'w').close()

        def summarizer(input, output, params):
            open(output.filename, 'w').close()

        RawImage = self.pipeline.file("{camera}_{scene}.jpg", name="RawImage")
        ProcessedImage = self.pipeline.file("{camera}_{scene}_processed.jpg", name="ProcessedImage")
        Summary = self.pipeline.file("summary.txt", name="Summary")
        self.pipeline.step(processor, RawImage, ProcessedImage)
        self.pipeline.step(summarizer, ProcessedImage.all(), Summary)
        success = self.pipeline.run()
        self.assertTrue(success)

        for camera in ["N1", "NP1", "NP2"]:
            for scene in ["0", "1"]:
                filename = "{0}_{1}_processed.jpg".format(camera,scene)
                full_path = os.path.join(self.test_path, filename)
                self.assertTrue(os.path.exists(full_path))
        full_path = os.path.join(self.test_path, "summary.txt")
        self.assertTrue(os.path.exists(full_path))

class DependencyGraphTest(TestCase):

    def test_topological_sort(self):
        graph = DependencyGraph()
        graph.add(0,2)
        graph.add(1,2)
        graph.add(1,3)
        graph.add(2,4)
        graph.add(3,4)
        graph.add(4,5)
        graph.add(4,6)

        topological = graph.topological_sort()

        # 3 doesn't depend on 0
        for i in [2,4,5,6]:
            self.assertLess(topological.index(0), topological.index(i))

        for i in range(2,7):
            self.assertLess(topological.index(1), topological.index(i))

        for i in range(4,7):
            self.assertLess(topological.index(2), topological.index(i))
            self.assertLess(topological.index(3), topological.index(i))

        for i in range(5,7):
            self.assertLess(topological.index(4), topological.index(i))

    def test_roots(self):
        graph = DependencyGraph()
        graph.add(0,2)
        graph.add(1,2)
        graph.add(1,3)
        graph.add(2,4)
        graph.add(3,4)
        graph.add(4,5)
        graph.add(4,6)

        roots = graph.roots()

        self.assertListEqual(roots, [0,1])

