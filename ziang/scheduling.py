

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


def generate_augmented_task_graph(task_graph,n_computers):
    """    
    Generate a new graph that describes computational task in the cluster setting
    Each node is associated with 

    """
    def nodename(orig_id,computer_idx):
        return "%s_%s"%(orig_id, computer_idx)


    aug = DirectedHypergraph()
    for node in task_graph.get_nodes():
        for i_computer in xrange(n_computers):
            # Node corresponding to having resource on a particular computer
            new_node = Node( nodename(node.name,i_computer), node.data.copy())            
            new_node.data["done"] = (i_computer==0) and node.data["done"]
            new_node.data["final"] = (i_computer==0) and node.data["final"]
            aug.add_node( new_node )
            # Edge corresponding to sending resource from i to j
            for j_computer in xrange(n_computers):
                if j_computer != i_computer:
                    name="send_%s_%i->%i"%(node.name,i_computer,j_computer)
                    aug.add_edge(HyperEdge(name, [nodename(node.name,i_computer)], [nodename(node.name,j_computer)], {"type":"comm","fromto":(i_computer,j_computer)}))
    for edge in task_graph.get_edges():
        for i_computer in xrange(n_computers):
            aug.add_edge(HyperEdge(edge.name, [nodename(node_id,i_computer) for node_id in edge.inputs] , [nodename(node_id,i_computer) for node_id in edge.outputs], edge.data.copy()))
    return aug

# def generate_schedule(task_graph):


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
        return self.worker2job[worker_id]

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

    def plan(self):
        raise NotImplementedError

def plan_with_ilp(tg):
    """
    Plan for a generic task hypergraph
    Each node and hyper-edge will have the "time" field added to its data dict.
    "time" will be set to None if the node's resource is never produced or the edge's job is never performed
    """

    import gurobipy
    m = gurobipy.Model()    
    edge2active = {}
    edge2start = {}
    node2done = {}
    t_total = m.addVar(vtype="C")
    t_max = 1000 # XXX
    for edge in tg.get_edges():
        edge2active[edge.name] = m.addVar(vtype="B")
        edge2start[edge.name] = m.addVar(vtype="C")
    for node in tg.get_nodes():
        node2done[node.name] = m.addVar(vtype="C")
    m.update()
    for node in tg.get_nodes():
        if node.data["final"]:
            m.addConstr(t_total >= node2done[node.name])
    for edge in tg.get_edges():
        for node_name in edge.inputs:
            m.addConstr(edge2start[edge.name] >= node2done[node_name])
        for node_name in edge.outputs:
            m.addConstr(node2done[node_name] >= edge2start[edge.name] + edge2active[edge.name]*edge.data["duration"])
        m.addConstr(edge2start[edge.name] >= (1-edge2active[edge.name])*t_max) # if edge is inactive, its time >= t_max
    m.setObjective(t_total)
    m.optimize()

    import IPython
    IPython.embed()


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

    cameras = ['N1', 'NP1', 'NP2']
    scenes = [0, 1]

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
    aug = generate_augmented_task_graph(tg,4)

    for edge in aug.get_edges():
        if edge.data["type"] == "comm":
            edge.data["duration"] = 1.0
        else:
            edge.data["duration"] = 10.0
    assert any(node.data["final"] for node in aug.get_nodes())
    plan_with_ilp(aug)


if __name__ == "__main__":
    test()