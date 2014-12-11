import tornado.ioloop
import tornado.web
from tornado.escape import json_encode
import base64

#import json
#config_file = open("config.json", 'r')
#config = json.load(config_file)
#config_file.close()

class RequestHandler(tornado.web.RequestHandler):
    master = None

    def post(self):
        worker_id = self.get_argument('worker_id', default=None)
        job_id = self.get_argument('job_id', default=None)
        result = self.get_argument('result', default=None)
        if result is not None:
            result = base64.b64decode(result)

        if job_id is not None:
            self.scheduler.process_completed_job(job_id, result)

        if not worker_id:
            response = {"status": "failed"}
            self.write(json_encode(response))
            return

        job_id, pickled = self.scheduler.get_job_for_worker(worker_id)
        if job_id is None:
            if self.scheduler.work_complete(worker_id):
                response = {"status": "finished"}
            else:
                response = {"status": "sleep"}
        else:
            pickled = base64.b64encode(pickled)
            response = {"job_id": job_id, "job": pickled}

        self.write(json_encode(response))


class Master(object):
    def __init__(self, scheduler, port=8888):
        RequestHandler.scheduler = scheduler
        self.port = port
        self.work_left = [1,2,3,4,5]
        self.scheduler = scheduler

    def check_pipeline_status(self):
        if self.scheduler.work_complete():
            print "stopping"
            tornado.ioloop.IOLoop.instance().stop()

    def run(self):
        application = tornado.web.Application([
            (r"/", RequestHandler),
        ])
        application.listen(self.port)
        cb = tornado.ioloop.PeriodicCallback(self.check_pipeline_status, 500)
        cb.start()
        print "in run"
        tornado.ioloop.IOLoop.instance().start()
        print "after start"
        jobs = self.scheduler.completed_jobs.values()
        return all([not job.failed for job in jobs]), self.scheduler.results

if __name__ == "__main__":
    master = Master()
    master.run()
