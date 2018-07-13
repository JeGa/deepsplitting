from collections import OrderedDict
import time


class Timing:
    def __init__(self):
        self.times = OrderedDict()
        self.t1 = 0
        self.current_key = None
        self.timer_started = False

    def __call__(self, name):
        self.current_key = name
        return self

    def __enter__(self):
        # self.t1 = time.process_time()
        self.t1 = time.perf_counter()

        self.timer_started = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.timer_started:
            # self.times[name] = time.process_time() - self.t1
            self.times[self.current_key] = time.perf_counter() - self.t1
            self.timer_started = False
            self.current_key = None
        else:
            raise UserWarning("Timer not started")
