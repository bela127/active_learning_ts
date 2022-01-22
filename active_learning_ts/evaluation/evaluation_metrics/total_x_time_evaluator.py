import time


class TotalXTimeEvaluator:

    def __init__(self):
        self.last_time = 0
        self.total_time = 0

    def signal_start(self):
        self.last_time = time.perf_counter()

    def signal_stop(self):
        self.total_time += (time.perf_counter() - self.last_time)

    def get_total(self):
        return self.total_time
