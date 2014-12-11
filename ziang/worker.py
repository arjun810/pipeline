import requests
import json
from time import sleep
import base64
import cloud.serialization.cloudpickle as _pickle
import traceback

#config_file = open("config.json", 'r')
#config = json.load(config_file)
#config_file.close()

class Worker(object):

    def __init__(self, master_host, master_port, id):
        self.finished = False
        self.master_url = "http://{0}:{1}/".format(master_host, master_port)
        self.id = id

    def report_result_and_request_job(self, job_id, pickled_result):
        if pickled_result is not None:
            pickled_result = base64.b64encode(pickled_result)
        data = {"job_id": job_id,
                "result": pickled_result,
                "worker_id": self.id}
        print data
        r = requests.post(self.master_url, data=data)
        print r
        print r.text
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
            return _pickle.loads(base64.b64decode(result["job"]))


    def run_job(self, job):
        if job is None:
            return None, None
        try:
            result = job.run()
            return job.id, _pickle.dumps(result)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            result_id = job.id
            traceback.print_exc()
            e.traceback = traceback.format_exc()
            return result_id, _pickle.dumps(e)

    def start(self):
        job_id = None
        pickled_result = None
        while not self.finished:
            job = self.report_result_and_request_job(job_id, pickled_result)
            job_id, pickled_result = self.run_job(job)
            sleep(0.05)

if __name__ == '__main__':
    w = Worker("127.0.0.1", 8888, 0)
    w.start()
