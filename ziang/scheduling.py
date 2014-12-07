

class HyperEdge(object):    

    def __init__(self,name,inputs,outputs,data):
        assert isinstance(name,str) and isinstance(inputs,list) and isinstance(outputs,list) and isinstance(data,dict)
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.data = data
    def __repr__(self):
        return "HyperEdge(%s,%s,%s,%s)"%(self.name,self.inputs,self.outputs,self.data)

class Node(object):

    def __init__(self,name,data):
        assert isinstance(name,str) and isinstance(data,dict)
        self.name = name
        self.data = data
    def __repr__(self):
        return "Node(%s,%s)"%(self.name,self.data)

class DirectedHypergraph(object):

    def __init__(self):
        self.edges = []
        self.nodes = []

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges
    
    def add_edge(self,edge):
        assert isinstance(edge,HyperEdge)
        self.edges.append(edge)

    def add_node(self,node):
        assert isinstance(node,Node)
        self.nodes.append(node)


    def get_incoming_edges(self,node_name):
        assert isinstance(node_name,str)
        # XXX inefficient
        return [edge for edge in self.edges if node_name in edge.outputs]

    def get_outgoing_edges(self,node_name):
        assert isinstance(node_name,str)
        # XXX inefficient
        return [edge for edge in self.edges if node_name in edge.inputs]

def generate_task_graph(dependency_graph):
    is_initial = lambda node: dependency_graph.graph.in_degree(node)==0
    is_final = lambda node: dependency_graph.graph.out_degree(node)==0
    from ziang import Job
    task_graph = DirectedHypergraph()
    for node in dependency_graph.graph.nodes_iter():
        if isinstance(node,Job):
            # Job corresponds to edge
            inputs = dependency_graph.parents(node)
            outputs = dependency_graph.children(node)
            task_graph.add_edge(HyperEdge(node.id,inputs,outputs,{"obj":node,"type":"job"}))            
        else:
            task_graph.add_node(Node(node,{"done":is_initial(node),"final":is_final(node)}))
    return task_graph


def get_frontier_edges(aug):
    """

    Find all edges that are not done but they are ready (all input nodes are done)
    """
    frontier = []
    for edge in aug.get_edges():
        if not edge.data["done"]:
            if all(input.data["done"] for input in edge.inputs):
                frontier.append(edge)
    return frontier


class ClusterScheduler(object):

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
        for node in get_frontier_edges(self.active):
            if node["worker"] == worker_id:
                return node["job"]
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

def djikstra(state_initial, is_goal, get_actions, successor, compute_score):
    pq = PriorityQueue()
    pq.push(0.0,state_initial)
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
                    next_score = compute_score(next_state)
                    pq.push(next_score,next_state)

from collections import namedtuple
import numpy as np

def plan_with_djikstra(tg,n_computers):
    State = namedtuple("State",["resources", "job_done", "comp_ready_time", "last_time","actions"])
    n_jobs = len(tg.edges)




    finals = [node.name for node in tg.nodes if len(tg.get_outgoing_edges(node.name))==0]

    is_goal = lambda state: all(final in state.resources[0] for final in finals)

    send_dur = lambda res:0.1
    job_dur = lambda jobidx:tg.edges[jobidx].data["dur"]

    Action = namedtuple("Action",["type","loc","jobidx","fromto","res"])

    def action_repr(action):
        if action.type=="c":
            return "%s @ %i"%(tg.edges[action.jobidx].name,action.loc)
        else:
            return "send %s @ %i->%i"%(action.res, action.fromto[0], action.fromto[1])

    def copy_resources(resources):
        return [s.copy() for s in resources]

    def get_actions1(state):
        # Computation actions
        for i_job in xrange(n_jobs):
            for i_c in xrange(n_computers):
                if all(input in state.resources[i_c] for input in tg.edges[i_job].inputs) and not all(output in state.resources[i_c] for output in tg.edges[i_job].outputs):
                    yield Action("c",i_c,i_job,None,None)
        for node in tg.nodes:
            node_name = node.name
            for i_c in xrange(n_computers):
                for j_c in xrange(n_computers):
                    if (i_c != j_c) and (node_name in state.resources[i_c]) and not (node_name in state.resources[j_c]):
                        yield Action("s",None,None,(i_c,j_c),node_name)
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
            new_resources[action.loc].update(tg.edges[action.jobidx].outputs)
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
        comp_lb = sum( edge.data['dur'] for (jobidx,edge) in enumerate(tg.edges) if not state.job_done[jobidx] )
        comm_lb = 0
        lb = (comp_lb+comm_lb) / n_computers        
        return state.last_time + lb*1.2

    state_initial = State([set() for _ in xrange(n_computers)], np.zeros(n_jobs,dtype=bool), np.zeros(n_computers,dtype='float32'), 0, []) #pylint: disable=E1101
    state_initial.resources[0].update(node.name for node in tg.nodes if node.data['done'])


    state_solution = djikstra(state_initial, is_goal,get_actions,successor,compute_score)
    print [action_repr(action) for action in state_solution.actions]



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
    scenes = [0,1]

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

    dg=pipeline.resource_job_graph
    tg = generate_task_graph(dg)
    assert any(node.data["final"] for node in tg.get_nodes())


    for edge in tg.get_edges():
        if edge.data["type"] == "comm":
            edge.data["dur"] = 1.0
        else:
            edge.data["dur"] = 10.0


    assert any(node.data["final"] for node in tg.get_nodes())
    # plan_with_ilp(tg)
    plan_with_djikstra(tg,2)


if __name__ == "__main__":
    test()