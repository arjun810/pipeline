import requests
import json
from time import sleep
import base64
import cloud.serialization.cloudpickle as _pickle
import traceback
import multiprocessing as mp
import Queue

#config_file = open("config.json", 'r')
#config = json.load(config_file)
#config_file.close()

def fetch(path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    cmd = "scp -o StrictHostKeyChecking=no -r ubuntu@ec2-54-184-142-40.us-west-2.compute.amazonaws.com:{0} {0}".format(path)
    os.system(cmd)

def _work(job_queue, result_queue, remote):
    while True:
        job = None
        try:
            job = _pickle.loads(base64.b64decode(job_queue.get()))
            if remote:
                if hasattr(type(job), package_path):
                    if not os.path.exists(job.package_path):
                        fetch(job.package_path)
                for input in job.task.input.values():
                    if isinstance(input, list):
                        input_list = input
                    else:
                        input_list = [input]
                    for file in input_list:
                        if not os.path.exists(file):
                            fetch(file)

            result = job.run()
            result_queue.put((job.id, _pickle.dumps(result)))
        except KeyboardInterrupt:
            pass
        except Exception as e:
            result_id = job.id if job is not None else None
            traceback.print_exc()
            e.traceback = traceback.format_exc()
            result_queue.put((result_id, _pickle.dumps(e)))

class Worker(object):

    def __init__(self, master_host, master_port, id, remote=False):
        self.finished = False
        self.master_url = "http://{0}:{1}/".format(master_host, master_port)
        self.id = id
        self.num_cores_to_use = 8
        self.jobs_running = 0
        self.remote = remote

    def setup_mp(self):
        self.job_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers = []
        for i in range(self.num_cores_to_use):
            worker = mp.Process(target=_work,
                                args=(self.job_queue, self.result_queue, self.remote))
            worker.start()
            self.workers.append(worker)

    def report_result_and_request_job(self, job_id, pickled_result):
        if pickled_result is not None:
            pickled_result = base64.b64encode(pickled_result)
        data = {"job_id": job_id,
                "result": pickled_result,
                "worker_id": self.id}
        r = requests.post(self.master_url, data=data)
        result = r.json()
        if "status" in result:
            status = result["status"]
            if status == "failed":
                return None
            if status == "finished":
                self.finished = True
                return None
            if status == "sleep":
                return None
        else:
            return result["job"]


    def process_completed_jobs(self):

        try:
            (job_id, result) = self.result_queue.get(timeout=1)
        except Queue.Empty:
            return

        job = self.report_result_and_request_job(job_id, result)

        if job is None:
            self.jobs_running -= 1
        else:
            self.job_queue.put(job)

    def cleanup(self):
        for worker in self.workers:
            worker.terminate()

    def start(self):
        self.setup_mp()

        while not self.finished or self.jobs_running > 0:
            if self.jobs_running < self.num_cores_to_use and not self.finished:
                job = self.report_result_and_request_job(None, None)
                if job is not None:
                    self.job_queue.put(job)
                    self.jobs_running += 1
            self.process_completed_jobs()
            sleep(0.01)
            print "{0}\r".format(self.jobs_running)

        self.cleanup()

if __name__ == '__main__':
    w = Worker("127.0.0.1", 8888, 0)
    w.start()
