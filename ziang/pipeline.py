import os
import traceback
import itertools
import multiprocessing as mp
import Queue
import inspect

#import dill as _pickle
#_pickle.dill._trace(False)
import cloud.serialization.cloudpickle as _pickle
import networkx as nx

class FileGroup(dict):

    def __key(self):
        return tuple((k,self[k]) for k in sorted(self))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()

    def __init__(self, dict):
        for k, v in dict.items():
            if isinstance(v, list):
                self[k] = FileList(v)
            else:
                self[k] = v

    def flatten(self):
        result = []
        for k, v in self.items():
            if isinstance(v, list):
                for file in v:
                    result.append(file)
            else:
                result.append(v)
        return result

class FileList(list):
  def __key(self):
    return tuple(sorted(self))
  def __hash__(self):
    return hash(self.__key())
  def __eq__(self, other):
    return self.__key() == other.__key()

class Task(object):

    def __init__(self, input, output, params):
        self.input = input
        self.output = output
        self.params = params
        self.validate()

    def validate(self):
        pass

    @classmethod
    def _prepare(cls, cde_path, output_dir):
        pass

    @classmethod
    def _validate(cls, input, output):
        for input_name, input_type in cls.input.items():
            if input_name not in input:
                err_msg = "Excepted {0} in input for task {1}"
                err_msg = err_msg.format(input_name, cls.__name__)
                raise ValueError(err_msg)
            if input_type == "filename":
                pass
            elif input_type == "filename_list":
                if not isinstance(input[input_name], list):
                    err_msg = "Excepted {0} to be a list for task {1}"
                    err_msg = err_msg.format(input_name, cls.__name__)
                    raise ValueError(err_msg)
            else:
                raise ValueError("Input type {0} not supported".format(input_type))

        for output_name, output_type in cls.output.items():
            if output_name not in output:
                err_msg = "Excepted {0} in output for task {1}"
                err_msg = err_msg.format(output_name, cls.__name__)
                raise ValueError(err_msg)
            if output_type == "filename":
                pass
            elif output_type == "filename_list":
                if not isinstance(output[output_name], list):
                    err_msg = "Excepted {0} to be a list for task {1}"
                    err_msg = err_msg.format(output_name, cls.__name__)
                    raise ValueError(err_msg)
            else:
                raise ValueError("Output type {0} not supported".format(output_type))

    def _run(self):
        self.run()

    def run(self):
        raise NotImplementedError("Task.run is abstract")

class BinaryTask(Task):

    executable = None
    package_command = None
    packaged = False

    @classmethod
    def _prepare(cls, cde_path, output_dir):
        if cls.packaged:
            return

        cls._package(cde_path, output_dir)

    @classmethod
    def _package(cls, cde_path, output_dir):

        cls.package_path = os.path.join(output_dir, cls.__name__)

        if cls.package_command is None:
            cmd = "{0} -o {1} {2} --help"
            cmd = cmd.format(cde_path, cls.package_path, cls.executable)
        else:
            cmd = "{0} -o {1} {2}"
            cmd = cmd.format(cde_path, cls.package_path, cls.package_command)

        os.system(cmd)
        cls.packaged = True

class Job(object):

    counter = itertools.count(0)

    def __init__(self, task):

        self.task = task

        self.task_name = self.determine_task_name()
        self.number = self.counter.next()
        self.done = False
        self.failed = False

        if hasattr(task, "num_cores"):
            self.num_cores = task.num_cores
        else:
            self.num_cores = 1

    def __repr__(self):
        return "Job: {0}".format(self.id)

    def determine_task_name(self):
        return "taskname"

    @property
    def id(self):
        return "{0}_{1}".format(self.task_name, self.number)

    def run(self):
        return self.task._run()

def _work(job_queue, result_queue):
    while True:
        job = None
        try:
            job = _pickle.loads(job_queue.get())
            # need something to create this task and mape arguments properly
            result = job.run()
            result_queue.put((job.id, _pickle.dumps(result)))
        except KeyboardInterrupt:
            pass
        except Exception as e:
            result_id = job.id if job is not None else None
            traceback.print_exc()
            e.traceback = traceback.format_exc()
            result_queue.put((result_id, _pickle.dumps(e)))

class Scheduler(object):

    def __init__(self, graph, num_cores_to_use):

        self.graph = graph
        self.num_cores_to_use = num_cores_to_use
        self.num_cores_in_use = 0

        self.candidate_jobs = {}
        self.pending_jobs = {}
        self.completed_jobs = {}
        self.results = {}

        for job in self.graph.roots():
            self.candidate_jobs[job.id] = job

        self.setup_mp()

    @property
    def num_cores_available(self):
        return self.num_cores_to_use - self.num_cores_in_use

    def setup_mp(self):
        self.job_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers = []
        for i in range(self.num_cores_to_use):
            worker = mp.Process(target=_work,
                                args=(self.job_queue, self.result_queue))
            worker.start()
            self.workers.append(worker)

    def schedulable(self, job):
        # If a job requests more cores than exist on the system, we should
        # schedule it anyways.
        all_cores_free = (self.num_cores_available == self.num_cores_to_use)
        return all_cores_free or job.num_cores <= self.num_cores_available

    def greedy_schedule(self):
        for job_id, job in self.candidate_jobs.items():
            if self.num_cores_available <= 0:
                break
            if self.schedulable(job):
                self.dispatch(job)

    def dispatch(self, job):
        try:
            self.job_queue.put(_pickle.dumps(job))
            self.pending_jobs[job.id] = job
        except Exception as e:
            self.results[job.id] = e
            job.failed = True
            self.completed_jobs[job.id] = job
            self.fail_downstream(job)

        del self.candidate_jobs[job.id]
        self.num_cores_in_use += job.num_cores

    def fail_downstream(self, job):
        for descendant in self.graph.descendants(job):
            self.results[descendant.id] = Exception("Failed due to upstream job {0}".format(job.id))
            descendant.failed = True
            self.completed_jobs[descendant.id] = descendant

    def process_completed_jobs(self):

        try:
            (job_id, result) = self.result_queue.get(timeout=1)
        except Queue.Empty:
            return
        result = _pickle.loads(result)
        self.results[job_id] = result

        job = self.pending_jobs[job_id]
        if isinstance(result, Exception):
            job.failed = True
        job.done = True

        del self.pending_jobs[job_id]
        self.completed_jobs[job_id] = job
        self.num_cores_in_use -= job.num_cores

        for child_job in self.graph.children(job):
            parents_done = all([j.done for j in self.graph.parents(child_job)])

            if parents_done:
                self.candidate_jobs[child_job.id] = child_job

    def run(self):
        #if len(self.candidate_jobs) == 0:
        #    raise Exception("no jobs to run..")
        while(len(self.completed_jobs) < len(self.graph)):
            self.greedy_schedule()
            self.process_completed_jobs()

        self.cleanup()

        jobs = self.completed_jobs.values()
        return all([not job.failed for job in jobs]), self.results

    def cleanup(self):
        for worker in self.workers:
            worker.terminate()

class DependencyGraph(object):

    def __init__(self):
        self.graph = nx.DiGraph()

    def __len__(self):
        return self.graph.number_of_nodes()

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

    def parents(self, node):
        return self.graph.predecessors(node)

    def children(self, node):
        return self.graph.successors(node)

    def ancestors(self, node):
        return nx.ancestors(self.graph, node)

    def descendants(self, node):
        return nx.descendants(self.graph, node)

    def topological_sort(self):
        return nx.topological_sort(self.graph)

    def roots(self):
        return [n for n,d in self.graph.in_degree().items() if d==0]

class Pipeline(object):

    def __init__(self):
        self.root_dir = ""
        self.jobs = {}
        self.globals = {}
        self.cde_path = "/home/arjun/CDE/cde"
        self.cde_output_dir = "/tmp/pipeline_cde"

        # Contains both resources and jobs
        self.resource_job_graph = DependencyGraph()

    def set_root(self, new_dir):
        filename = inspect.stack()[-1][1]
        script_path = os.path.dirname(os.path.abspath(filename))
        if new_dir[0] == "/":
            pass
        elif new_dir[0] == "~":
            new_dir = os.path.abspath(os.path.expanduser(new_dir))
        else:
            new_dir = os.path.join(script_path, new_dir)

        self.root_dir = new_dir

    def preprocess_filenames(self, file_group):
        for name, value in file_group.items():
            if isinstance(value, list):
                new_list = []
                for filename in value:
                    new_list.append(os.path.join(self.root_dir, filename))
                file_group[name] = new_list
            else:
                file_group[name] = os.path.join(self.root_dir, value)
        return FileGroup(file_group)

    def add_task(self, task_type, input, output, **kwargs):
        params = self.globals.copy()
        params.update(kwargs)
        input = self.preprocess_filenames(input)
        output = self.preprocess_filenames(output)
        task = self.build_task(task_type, input, output, params)
        job = Job(task)

        for filename in input.flatten():
            self.resource_job_graph.add(filename, job)

        for filename in output.flatten():
            self.resource_job_graph.add(job, filename)

        self.jobs[job.id] = job

    def build_task(self, task_type, input, output, params):
        task_type._prepare(self.cde_path, self.cde_output_dir)
        task_type._validate(input, output)
        task = task_type(input, output, params)
        return task

    def compute_job_graph(self):
        topological_joint = self.resource_job_graph.topological_sort()
        topological_jobs = itertools.ifilter(lambda x: isinstance(x, Job),
                                             topological_joint)
        graph = DependencyGraph()
        for job in topological_jobs:
            child_jobs = []
            for resource in self.resource_job_graph.children(job):
                child_jobs.extend(self.resource_job_graph.children(resource))
            graph.add(job, child_jobs)
        self.job_graph = graph

    def run(self, num_cores_to_use=None):
        if num_cores_to_use is None:
            num_cores_to_use = mp.cpu_count()*2
        self.compute_job_graph()
        scheduler = Scheduler(self.job_graph, num_cores_to_use)
        success, results = scheduler.run()
        if not success:
            failed_jobs = filter(lambda x: x.failed, self.jobs.values())
            print "-"*80
            for job in failed_jobs:
                print "Job failed: {0}".format(job.id)
                print self.job_spec(job)
                print "Exception: "
                print results[job.id]
                print results[job.id].traceback
                print "-"*80
        return success

    def job_spec(self, job):
        inputs = self.resource_job_graph.parents(job)
        outputs = self.resource_job_graph.children(job)
        spec = "Inputs:\n"
        for input in inputs:
            spec += "\t{0}\n".format(input.filename)
        spec += "Outputs:\n"
        for output in outputs:
            spec += "\t{0}\n".format(output.filename)
        return spec
