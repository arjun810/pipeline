import os
import itertools
import types

import lucidity

class Resource(object):
    pass

class FilePattern(object):

    def __init__(self, root_dir, pattern):
        if os.path.isabs(pattern):
            raise Exception("Patterns should be relative to the root directory.")
        self.root_dir = root_dir
        self.pattern = os.path.join(root_dir, pattern)

        # lucidity requires naming each Template. As a hack for now, we'll just
        # name the template the pattern itself, to ensure there are no
        # conflicts.
        self.template = lucidity.Template(self.pattern, self.pattern)
        self.variables = self.template.keys()

    def pattern_root(self):

        # Note: this will need to use splitdrive to work on Windows
        components = self.pattern.split('/')
        pattern_root = []
        for component in components:
            if "{" in component:
                break
            pattern_root.append(component)

        return "/".join(pattern_root)

    def discover(self):
        root = self.pattern_root()
        candidates = [os.path.join(dp, f) for dp, dn, fn in os.walk(root) for f in fn]
        self.filenames = []
        self.instances = []

        for filename in candidates:
            try:
                instance = self.template.parse(filename)
                instance['filename'] = filename
                self.filenames.append(filename)
                self.instances.append(instance)
            except lucidity.ParseError:
                pass

    def format(self, data):
        return self.template.format(data)

class ResourceIterator(object):

    def __init__(self, source):
        self.source = source
        self.conditions = {}

    def __iter__(self):
        return self.each()

    def each(self):
        for v in self.source.values():
            if self.accept(v):
                yield v

    def all(self):
        for v in [list(self.each())]:
            yield v

    def first(self):
        v = next(self.each(), None)
        if v is None:
            raise ValueError("No values found matching conditions {0}.".format(self.conditions))
        return v

    def where(self, conditions):
        valid_keys = self.source.valid_keys()
        for k, v in conditions.items():
            if k not in valid_keys:
                message = "Invalid key: {0}. ({1})".format(k, valid_keys)
                raise ValueError(message)
            self.conditions[k] = str(v)
        return self

    def accept(self, item):
        for k, v in self.conditions.items():
            if item[k] != v:
                return False
        return True


class FileResource(Resource):

    def __init__(self, root_dir, pattern):
        self.pattern = FilePattern(root_dir, pattern)
        self.variables = self.pattern.variables
        self.pattern.discover()

    def __iter__(self):
        return ResourceIterator(self).each()

    def each(self):
        return ResourceIterator(self).each()

    def all(self):
        return ResourceIterator(self).all()

    def where(self, **kwargs):
        return ResourceIterator(self).where(kwargs)

    def valid_keys(self):
        return self.variables

    def values(self):
        return self.pattern.instances

    def build(self, input):

        if not isinstance(input, list):
            # Terrible. The inputs/outputs need to be factored into objects...
            input_variables = set(input.keys())
            input_variables.remove("filename")

            # If input is a single element, then the built resource's
            # variables should be equal to the input resource's variables
            if input_variables != set(self.variables):
                raise ValueError("Expected variables {0}. Got {1}".format(input_variables,
                                                                          self.variables))
            output = {}
            for k, v in input.items():
                output[k] = v
        else:
            # If we have a list of inputs, then the built resource must be a
            # reduction and have fewer keys than the inputs. Any keys that the
            # built resource has must be identical across all inputs.
            set_variables = set(self.variables)
            output = {}

            # Initialize output using the first input
            for k in self.variables:
                if k not in input[0]:
                    raise ValueError("Expected {0} to be a variable of {1}".format(k, input[0]))
                output[k] = input[0][k]

            for item in input:
                input_variables = set(item.keys())
                input_variables.remove("filename")
                if not set_variables.issubset(input_variables):
                    raise ValueError("Expected {1} to be a subset of {0}".format(input_variables,
                                                                                 self.variables))
                for k in self.variables:
                    if output[k] != item[k]:
                        raise ValueError("Expected {0}[{1}] to have value {2}".format(item, k, output[k]))
                    pass

        output["filename"] = self.pattern.format(output)
        return output






class Pipeline(object):

    def __init__(self):
        self.current_dir = None
        self.queue = []

    def file(self, pattern):
        if self.current_dir is None:
            raise Exception("Pipeline: must chdir before creating file resource.")
        return FileResource(self.current_dir, pattern)

    def chdir(self, new_dir):
        if new_dir[0] == "/":
            pass
        elif new_dir[0] == ".":
            print "Warning, using relative path based on *where python is invoked*"
            new_dir = os.path.abspath(os.path.expanduser(new_dir))
        else:
            print "Warning, using relative path based on *where python is invoked*"
            new_dir = "./" + new_dir
            new_dir = os.path.abspath(os.path.expanduser(new_dir))

        self.current_dir = new_dir

    def append_task(self, task, input, output):
        self.queue.append((task, input, output))

    def step_lists(self, task, inputs, outputs):
        if len(inputs) == len(outputs):
            for i, o in zip(inputs, outputs):
                self.append_task(task, i, o)
        else:
            raise ValueError("Input and output lists must be of same length."
                                "input: {0}, output: {1}".format(len(inputs),
                                                                 len(outputs)))


    def step_gens(self, task, inputs, output_resource):
        for input in inputs:
            output = output_resource.build(input)
            self.append_task(task, input, output)

    def step(self, task, inputs, output_resource):

        if not isinstance(inputs, types.GeneratorType):
            # Case 1: inputs is not a generator
            return self.step_lists(task, inputs, output_resource)
        else:
            # Case 2: inputs is a generator
            return self.step_gens(task, inputs, output_resource)
