import os
import itertools
import multiprocessing as mp

import networkx as nx
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
        self.values = self.source.instances

    def __iter__(self):
        return self.each()

    def each(self):
        for v in self.values:
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

    def group_by(self, key):
        # TODO as stated elsewhere, values should probably be something other
        # than dicts, so that this doesn't need to be such a hack.
        keyfunc = lambda x: getattr(x,key)
        values = sorted(self.values, key=keyfunc)
        grouped = itertools.groupby(values, key=keyfunc)
        self.values = [list(g[1]) for g in grouped]
        return self

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
            if getattr(item, k) != v:
                return False
        return True

class FileResourceMeta(type):

    def __init__(cls, name, bases, dct):
        super(FileResourceMeta, cls).__init__(name, bases, dct)
        if cls.pattern is not None:
            cls.discover()

    def discover(cls):
        cls.variables = cls.pattern.variables
        cls.pattern.discover()
        cls.instances = []
        for pattern_instance in cls.pattern.instances:
            instance = cls(**pattern_instance)
            instance.exists = True
            cls.instances.append(instance)

    def __iter__(cls):
        return ResourceIterator(cls).each()

    def each(cls):
        return ResourceIterator(cls).each()

    def all(cls):
        return ResourceIterator(cls).all()

    def where(cls, **kwargs):
        return ResourceIterator(cls).where(kwargs)

    def group_by(cls, key):
        return ResourceIterator(cls).group_by(key)

    def valid_keys(cls):
        return cls.variables

    def existing(cls):
        return filter(lambda x: x.exists, cls.instances)

    def pending(cls):
        return filter(lambda x: not x.exists, cls.instances)

    def build(cls, input):

        output_attrs = {}
        if not isinstance(input, list) and not isinstance(input, tuple):
            # Terrible. The inputs/outputs need to be factored into objects...
            input_variables = set(input.variables)

            # If input is a single element, then the built resource's
            # variables should be equal to the input resource's variables
            if input_variables != set(cls.variables):
                raise ValueError("Expected variables {0}. Got {1}".format(input_variables,
                                                                          cls.variables))
            for k in input.variables:
                output_attrs[k] = getattr(input, k)
        else:
            # If we have a list of inputs, then the built resource must be a
            # reduction and have fewer keys than the inputs. Any keys that the
            # built resource has must be identical across all inputs.
            set_variables = set(cls.variables)

            # Initialize output using the first input
            for k in cls.variables:
                if k not in input[0].variables:
                    raise ValueError("Expected {0} to be a variable of {1}".format(k, input[0]))
                output_attrs[k] = getattr(input[0], k)

            for item in input:
                input_variables = set(item.variables)
                if not set_variables.issubset(input_variables):
                    raise ValueError("Expected {1} to be a subset of {0}".format(input_variables,
                                                                                 cls.variables))
                for k in cls.variables:
                    if output_attrs[k] != getattr(item, k):
                        raise ValueError("Expected {0}[{1}] to have value {2}".format(item, k, output_attrs[k]))
                    pass

        output_attrs["filename"] = cls.pattern.format(output_attrs)
        new_instance = cls(**output_attrs)
        cls.instances.append(new_instance)
        return new_instance


class FileResource(Resource):
    __metaclass__ = FileResourceMeta
    pattern = None

    def __init__(self, **kwargs):
        self.exists = False
        allowed_fields = self.variables + ["filename"]
        for k, v in kwargs.items():
            if k not in allowed_fields:
                raise ValueError("Valid values are {0}. Got {1}.".format(self.variables, k))
            setattr(self, k, str(v))

    @property
    def id(self):
        base = self.__class__.__name__
        attrs = {}
        for k in self.variables:
            attrs[k] = getattr(self, k)
        return "{0}_{1}".format(base, attrs)

class Job(object):

    counter = itertools.count(0)

    def __init__(self, task, inputs, outputs):
        self.task = task
        self.inputs = inputs
        self.outputs = outputs
        self.task_name = self.determine_task_name()
        self.number = self.counter.next()

    def __repr__(self):
        return "Job: {0}".format(self.id)

    def determine_task_name(self):
        return "taskname"

    @property
    def id(self):
        return "{0}_{1}".format(self.task_name, self.number)

class Stage(object):

    def __init__(self, name):
        self.name = name
        self.job_ids = []

    def append(self, job_id):
        self.job_ids.append(job_id)

class DependencyGraph(object):

    def __init__(self):
        self.graph = nx.DiGraph()

    def add(self, parents, children):

        if not isinstance(parents, list):
            parents = [parents]

        if not isinstance(children, list):
            children = [children]

        for parent in parents:
            self.graph.add_node(parent)
            for child in children:
                self.graph.add_node(child)
                self.graph.add_edge(parent, child)

    def children(self, node):
        return self.graph.successors(node)

    def topological_sort(self):
        return nx.topological_sort(self.graph)

class Pipeline(object):

    def __init__(self, num_processes=1):
        self.current_dir = None
        self.stages = []
        self.jobs = {}
        self.num_processes = num_processes
        self.cores_in_use = 0
        self.queue = []

        # Contains both resources and jobs
        self.joint_dependency_graph = DependencyGraph()

        #self.dispatcher = MultiprocessingDispatcher()

    def file(self, template, name=None):
        if self.current_dir is None:
            raise Exception("Pipeline: must chdir before creating file resource.")
        pattern = FilePattern(self.current_dir, template)
        dct = {"pattern": pattern}
        if name is None:
            name = template
        resource = FileResourceMeta(name, (FileResource,), dct)
        return resource

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

    def determine_stage_name(self, task):
        return "stagename"

    def step(self, task, inputs, output_resource):
        #stage_name = self.get_stage_name(task)
        #stage = Stage(stage_name)
        for i, input in enumerate(inputs):
            output = output_resource.build(input)
            job = Job(task, input, output)
            self.joint_dependency_graph.add(input, job)
            self.joint_dependency_graph.add(job, output)
            self.jobs[job.id] = job
            self.queue.append(job)
        #self.stages.append(stage)

    def compute_job_graph(self):
        topological_joint = self.joint_dependency_graph.topological_sort()
        topological_jobs = itertools.ifilter(lambda x: isinstance(x, Job),
                                             topological_joint)
        graph = DependencyGraph()
        for job in topological_jobs:
            for resource in self.joint_dependency_graph.children(job):
                child_jobs = self.joint_dependency_graph.children(resource)
                for child_job in child_jobs:
                    graph.add(job, child_job)
        self.job_graph = graph


    def run(self):
        self.compute_job_graph()

    #    for stage in self.stages:
    #        self.dispatcher

    #    self.workers = []
    #    self.job_queue = []
    #    for i in range(self.num_processes):
    #        worker = mp.Process(target=_work, args=(task_queue
    #    = multiprocessing.Pool(self.num_processes)
    #    for task in self.queue:
    #        print "here"
    #        self.execute_task(task)
    #    self.pool.join()
