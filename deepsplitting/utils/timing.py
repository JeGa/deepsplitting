from collections import OrderedDict
import time


class Timing:
    def __init__(self):
        self.times = OrderedDict()
        self.t1 = 0
        self.started = False

    def start(self):
        self.started = True
        # self.t1 = time.process_time()
        self.t1 = time.perf_counter()

    def stop(self, name):
        if self.started:
            # self.times[name] = time.process_time() - self.t1
            self.times[name] = time.perf_counter() - self.t1
            self.started = False
        else:
            raise UserWarning("Timer not started")
