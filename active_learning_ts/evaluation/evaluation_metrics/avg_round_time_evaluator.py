import time

from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric


class AvgRoundTimeEvaluator(EvaluationMetric):
    """
    Evaluation Metric. Evaluates the average time required for training. Evaluation a list of the average round times

    e.g. evaluation: [avg time for round 1, avg time for round 1-2, ..., avg time for round 1-n]
    """
    def __init__(self):
        self.round_number = 0
        self.start_time = 0
        self.total_time = 0
        self.averages = []

    def signal_round_start(self):
        self.start_time = time.perf_counter()

    def signal_round_stop(self):
        self.round_number += 1
        self.total_time += time.perf_counter() - self.start_time

    def eval(self):
        self.averages.append(self.total_time / self.round_number)

    def get_evaluation(self):
        return self.averages
