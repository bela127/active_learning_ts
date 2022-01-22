from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.evaluation.evaluation_metrics.total_x_time_evaluator import TotalXTimeEvaluator


class TotalQueryTimeEvaluator(EvaluationMetric):
    def __init__(self):
        self.timer = TotalXTimeEvaluator()

    def signal_query_start(self):
        self.timer.signal_start()

    def signal_query_stop(self):
        self.timer.signal_stop()

    def get_evaluation(self):
        return self.timer.get_total()
