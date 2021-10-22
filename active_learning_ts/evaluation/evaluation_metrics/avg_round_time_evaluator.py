from typing import Protocol
import time

from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric


class AvgRoundTimeEvaluator(EvaluationMetric):
    def __init__(self):
        self.round_number = 0
        self.start_time = time.clock()
        self.averages = []

    def eval(self):
        self.round_number += 1
        self.averages.append((time.clock() - self.start_time) / self.round_number)

    def get_evaluation(self):
        return self.averages
