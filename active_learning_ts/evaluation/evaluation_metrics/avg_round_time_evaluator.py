import time

from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric


class AvgRoundTimeEvaluator(EvaluationMetric):
    """
    Evaluation Metric. Evaluates the average time required for training. Evaluation a list of the average round times

    e.g. evaluation: [avg time for round 1, avg time for round 1-2, ..., avg time for round 1-n]
    """
    def __init__(self):
        self.round_number = 0
        self.start_time = time.perf_counter()
        self.averages = []

    def eval(self):
        self.round_number += 1
        self.averages.append((time.perf_counter() - self.start_time) / self.round_number)

    def get_evaluation(self):
        return self.averages
