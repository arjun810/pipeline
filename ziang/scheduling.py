import networkx as nx

class Job(object):
    def __init__(self,name,inputs,outputs,data):
        assert isinstance(name,str) and isinstance(inputs,list) and isinstance(outputs,list) and isinstance(data,dict)
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.data = data
    def __repr__(self):
        return "Job(%s,%s,%s,%s)"%(self.name,self.inputs,self.outputs,self.data)

class Resource(object):

    def __init__(self,name,data):
        assert isinstance(name,str) and isinstance(data,dict)
        self.name = name
        self.data = data
    def __repr__(self):
        return "Resource(%s,%s)"%(self.name,self.data)

class TaskHypergraph(object):

    """
    Data structure with a set of jobs and a set of resources,
    which is convenient for planning algorithms
    """

    def __init__(self):
        self.jobs = []
        self.resources = []
        self.g = nx.DiGraph()

    def get_resources(self):
        return self.resources

    def get_jobs(self):
        return self.jobs

    def add_job(self,job):
        assert isinstance(job,Job)
        self.jobs.append(job)
        self.g.add_node(job.name,obj=job,type="job")
        for res in job.inputs:
            self.g.add_edge(job.name,res)
        for res in job.outputs:
            self.g.add_edge(res,job.name)

    def add_resource(self,resource):
        assert isinstance(resource,Resource)
        self.resources.append(resource)
        self.g.add_node(resource.name,obj=resource, type="resource")

    def get_resource(self,resource_name):
        return self.g.node[resource_name]["obj"]

    def get_job(self, job_name):
        return self.g.node[job_name]["obj"]

    def get_incoming_jobs(self,resource_name):
        return self.g.node[resource_name].in_jobs()

    def get_outgoing_jobs(self,resource_name):
        return self.g.node[resource_name].out_jobs()

    def jobs_toposorted(self):
        return [resource for resource in nx.topological_sort(self.g,reverse=True) if self.g.node[resource]["type"] == "job"]


def generate_task_graph(dependency_graph):
    """
    Convert pipeline.DependencyGraph into TaskHypergraph
    """
    is_initial = lambda resource: dependency_graph.graph.in_degree(resource)==0
    is_final = lambda resource: dependency_graph.graph.out_degree(resource)==0
    import ziang
    task_graph = TaskHypergraph()
    for resource in dependency_graph.graph.nodes_iter():
        if isinstance(resource,ziang.Job):
            # Job corresponds to job
            inputs = dependency_graph.parents(resource)
            outputs = dependency_graph.children(resource)
            task_graph.add_job(Job(resource.id,inputs,outputs,{"obj":resource,"type":"job"}))
        else:
            task_graph.add_resource(Resource(resource,{"done":is_initial(resource),"final":is_final(resource)}))
    return task_graph


def get_frontier_jobs(aug):
    """
    Find all jobs that are not done but they are ready (all input resources are done)
    """
    frontier = []
    for job in aug.get_jobs():
        if not job.data["done"]:
            if all(input.data["done"] for input in job.inputs):
                frontier.append(job)
    return frontier


class ClusterScheduler(object):
    """
    Not currently functional
    """
    def __init__(self, aug):
        self.aug = aug
        self.active = None
        self.job2worker = {}
        self.worker2job = {}

    def handle_worker(self, worker_id):
        # What job was that worker doing (possibly None)
        job_id = self.get_current_job(worker_id)
        # Mark it as done
        if job_id is not None:
            self.mark_as_done(job_id)
        # Is there a new job ready for him?
        new_job_id = self.find_job(worker_id)
        if new_job_id is not None:
            self.mark_as_assigned(worker_id, new_job_id)
        return new_job_id

    def get_current_job(self, worker_id):
        return self.worker2job.get(worker_id)

    def mark_as_done(self, job_id):
        worker_id = self.job2worker[job_id]
        del self.worker2job[worker_id]
        del self.job2worker[job_id]

    def mark_as_assigned(self, job_id, worker_id):
        self.job2worker[job_id] = worker_id
        self.worker2job[worker_id] = job_id

    def find_job(self,worker_id):
        """
        See if worker is assigned to task that is on the frontier of active task graph (self.active).
        """
        assert self.plan_is_ready()
        for resource in get_frontier_jobs(self.active):
            if resource["worker"] == worker_id:
                return resource["job"]
        return None

    def plan_is_ready(self):
        return self.active is not None


import heapq
import itertools
class PriorityQueue(object):
    def __init__(self):
        self.h = []
        self.count = itertools.count()
    def push(self,score,item):
        assert isinstance(score,float)
        heapq.heappush(self.h,(score,self.count.next(),item))
    def pop(self):
        return heapq.heappop(self.h)

def djikstra(state_initial, is_goal, get_actions, successor, compute_score, compute_hash=None):
    pq = PriorityQueue()
    pq.push(0.0,state_initial)
    closed = set()
    while True:
        try:
            score,_,state = pq.pop()
        except IndexError:
            print "failed to find path"
            break
        if is_goal(state):
            return state
        else:
            for action in get_actions(state):
                next_state = successor(state,action)
                if next_state is not None:
                    if compute_hash is not None:
                        h = compute_hash(next_state)
                        if h in closed:
                            continue
                        else:
                            closed.add(h)
                    next_score = compute_score(next_state)
                    pq.push(next_score,next_state)


def plan_with_djikstra(tg,n_computers):
    from collections import namedtuple
    import numpy as np
    import hashlib
    State = namedtuple("State",["resources", "job_done", "comp_ready_time", "last_time","actions"])
    n_jobs = len(tg.jobs)




    finals = [resource.name for resource in tg.get_resources() if len(tg.get_outgoing_jobs(resource.name))==0]

    is_goal = lambda state: all(final in state.resources[0] for final in finals)

    send_dur = lambda res:0.1
    job_dur = lambda jobidx:tg.jobs[jobidx].data["job_dur"]

    Action = namedtuple("Action",["type","loc","jobidx","fromto","res"])

    def action_repr(action):
        if action.type=="c":
            return "%s @ %i"%(tg.jobs[action.jobidx].name,action.loc)
        else:
            return "send %s @ %i->%i"%(action.res, action.fromto[0], action.fromto[1])

    def copy_resources(resources):
        return [s.copy() for s in resources]

    def compute_hash(state):
        h = hashlib.md5()
        for (i,res_set) in enumerate(state.resources):
            h.update(str(i))
            for res in res_set:
                h.update(res)
        h.update(state.job_done)
        h.update(state.comp_ready_time)
        h.update(state.last_time)
        return h.hexdigest()

    def get_actions1(state):
        # Computation actions
        for i_job in xrange(n_jobs):
            for i_c in xrange(n_computers):
                if all(input in state.resources[i_c] for input in tg.jobs[i_job].inputs) and not all(output in state.resources[i_c] for output in tg.jobs[i_job].outputs):
                    yield Action("c",i_c,i_job,None,None)
        for resource in tg.get_resources():
            resource_name = resource.name
            for i_c in xrange(n_computers):
                for j_c in xrange(n_computers):
                    if (i_c != j_c) and (resource_name in state.resources[i_c]) and not (resource_name in state.resources[j_c]):
                        yield Action("s",None,None,(i_c,j_c),resource_name)
    def get_actions(state):
        for a in get_actions1(state):
            # print action_repr(a), state.resources
            yield a

    def successor(state,action):

        new_resources = copy_resources(state.resources)
        new_job_done = state.job_done.copy()
        new_comp_ready_time = state.comp_ready_time.copy()

        if action.type == "c":
            start_time = state.comp_ready_time[action.loc]
            if start_time < state.last_time: return None
            new_resources[action.loc].update(tg.jobs[action.jobidx].outputs)
            new_job_done[action.jobidx] = True
            new_comp_ready_time[action.loc] = start_time + job_dur(action.jobidx)
            new_last_time = start_time
        else:
            fro,to = action.fromto
            start_time = max(state.comp_ready_time[fro],state.comp_ready_time[to])
            if start_time < state.last_time: return None
            assert fro!=to
            new_resources[to].add(action.res)
            new_comp_ready_time[fro] = new_comp_ready_time[to] = start_time + send_dur(action.res)
            new_last_time = start_time

        new_actions = state.actions + [action]
        return State(new_resources, new_job_done, new_comp_ready_time, new_last_time, new_actions)


    def compute_score(state):
        comp_lb = sum( job.data['dur'] for (jobidx,job) in enumerate(tg.jobs) if not state.job_done[jobidx] )
        comm_lb = 0
        lb = (comp_lb+comm_lb) / n_computers
        return state.last_time + lb*1.2

    state_initial = State([set() for _ in xrange(n_computers)], np.zeros(n_jobs,dtype=bool), np.zeros(n_computers,dtype='float32'), 0, []) #pylint: disable=E1101
    state_initial.resources[0].update(resource.name for resource in tg.resources if resource.data['done'])


    state_solution = djikstra(state_initial, is_goal,get_actions,successor,compute_score,compute_hash)
    print [action_repr(action) for action in state_solution.actions]

def hill_climb(initial_soln,compute_move,compute_cost,n_trials):
    """
    Hill climbing algorithm
    Iteratively perturb the solution and see if the cost improves


    Inputs
    -------
    initial_soln : solution data structure

    compute_move : function: soln -> soln, which generates a new random solution from the old one
    compute_cost : function: soln -> float, which gives the cost of a solution
    n_trials : how many random solutions to try out
    """
    current_cost = compute_cost(initial_soln)
    current_soln = initial_soln
    for i_trial in xrange(n_trials):
        if i_trial % 100 == 0:
            print "trial %i cost %f"%(i_trial,current_cost)
        candidate_soln = compute_move(current_soln)
        candidate_cost = compute_cost(candidate_soln)
        if candidate_cost < current_cost:
            current_cost = candidate_cost
            current_soln = candidate_soln
    return current_cost, current_soln

import random
def plan_with_hill_climb(tg,n_computers,n_trials=3000,initialize=None):
    """
    Compute an assignment of tasks to computers with a hill combing algorithm
    """

    if initialize is None:
        initial_soln = {job.name:0 for job in tg.get_jobs()}
    else:
        initial_soln = {job.name:initialize[job.name] for job in tg.get_jobs()}

    job_iter = itertools.cycle([job.name for job in tg.get_jobs()])
    def compute_move(soln):
        new_soln = soln.copy()
        changejob = job_iter.next()
        # changejob = random.choice(soln.keys())
        # newval = random integer in [0,n_computers-1] != soln[idx]
        newval = random.randint(0,n_computers-2)
        if newval >= soln[changejob]: newval += 1
        new_soln[changejob] = newval
        return new_soln

    compute_cost_lowerbound = lambda soln : compute_cost_with_assignment(tg, soln, n_computers)

    return hill_climb(initial_soln, compute_move, compute_cost_lowerbound,n_trials)

def compute_cost_naive(tg, n_computers,return_job2loc=False):
    """
    Compute cost of naive controller ignoring bandwidth limits and NOT doing concurrent computation & download
    """
    resource2creationtimeloc = {resource.name:(0.0,0) for resource in tg.resources if resource.data['done']}
    computer_queue = PriorityQueue()
    job2loc = {}
    t_total=0
    for i in xrange(n_computers):
        computer_queue.push(0.0,i)

    for job_name in tg.jobs_toposorted():
        job = tg.get_job(job_name)
        t_startjob,_,job_loc = computer_queue.pop()
        for resource_name in job.inputs:
            t_resource,i_loc = resource2creationtimeloc[resource_name]
            t_startjob = max(t_startjob,t_resource + (i_loc != job_loc)*tg.get_resource(resource_name).data["send_dur"] )
        t_finishjob = t_startjob + job.data["job_dur"]
        for resource_name in job.outputs:
            resource2creationtimeloc[resource_name] = (t_finishjob,job_loc)

        t_total = max(t_total, t_finishjob)
        computer_queue.push(t_finishjob,job_loc)
        job2loc[job.name] = job_loc


    if return_job2loc:
        return t_total,job2loc
    else:
        return t_total

def compute_cost_with_assignment(tg, job2loc, n_computers):
    resource2creationtimeloc = {resource.name:(0.0,0) for resource in tg.resources if resource.data['done']}
    computer2nextfreetime = [0 for _ in xrange(n_computers)]
    t_total = 0
    for job_name in tg.jobs_toposorted():
        job = tg.get_job(job_name)
        job_loc = job2loc[job_name]
        t_startjob = computer2nextfreetime[job_loc]
        for resource_name in job.inputs:
            t_resource,i_loc = resource2creationtimeloc[resource_name]
            t_startjob = max(t_startjob,t_resource + (i_loc != job_loc)*tg.get_resource(resource_name).data["send_dur"] )
        t_finishjob = t_startjob + job.data["job_dur"]
        computer2nextfreetime[job_loc] = t_finishjob
        for resource_name in job.outputs:
            resource2creationtimeloc[resource_name] = (t_finishjob,job_loc)

        t_total = max(t_total, t_finishjob)

    return t_total




def test():
    import sys
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

    cameras = ['A', 'B', 'C']
    scenes = range(10)

    for camera in cameras:
        for scene in scenes:
            input = {'image': "{0}_{1}.jpg".format(camera, scene)}
            output = {'image': "{0}_{1}_processed.jpg".format(camera, scene)}
            pipeline.add_task(ImageProcessor, input, output)

            input=output
            output={'image': "{0}_{1}_processed2.jpg".format(camera, scene)}
            pipeline.add_task(ImageProcessor, input, output)


    input = {}
    input['images'] = []
    for camera in cameras:
        for scene in scenes:
            input['images'].append("{0}_{1}_processed.jpg".format(camera, scene))
    output = {'summary': "summary.txt"}
    pipeline.add_task(ImageSummarizer, input, output)

    dg=pipeline.resource_job_graph
    tg = generate_task_graph(dg)
    assert any(resource.data["final"] for resource in tg.get_resources())


    for job in tg.get_jobs():
        job.data["job_dur"] = 10.0

    for resource in tg.get_resources():
        resource.data["send_dur"] = 3.0


    assert any(resource.data["final"] for resource in tg.get_resources())
    # plan_with_ilp(tg)
    # plan_with_djikstra(tg,2)
    n_computers=15
    cost_naive,job2loc_naive = compute_cost_naive(tg,n_computers,return_job2loc=True)
    print "TIME FROM NAIVE PLANNER", cost_naive
    assert cost_naive == compute_cost_with_assignment(tg, job2loc_naive, n_computers)

    print "RUNNING HILL-CLIMBING PLANNER INITIALIZED FROM NAIVE PLANNER"
    cost_hc,job2loc_hc = plan_with_hill_climb(tg,n_computers,initialize=job2loc_naive)
    print "TIME FROM HILL-CLIMBING PLANNER INITIALIZED WITH NAIVE",cost_hc
    assert cost_hc == compute_cost_with_assignment(tg, job2loc_hc, n_computers)

    print "RUNNING HILL-CLIMBING PLANNER"
    cost_hc,job2loc_hc = plan_with_hill_climb(tg,n_computers)
    print "TIME FROM HILL-CLIMBING PLANNER",cost_hc
    assert cost_hc == compute_cost_with_assignment(tg, job2loc_hc, n_computers)



if __name__ == "__main__":
    test()
