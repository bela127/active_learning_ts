from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.evaluation.evaluation_metrics.total_x_time_evaluator import TotalXTimeEvaluator


class TotalKnowledgeDiscoveryTimeEvaluator(EvaluationMetric):
    def __init__(self):
        self.timer = TotalXTimeEvaluator()

    def signal_knowledge_discovery_start(self):
        self.timer.signal_start()

    def signal_knowledge_discovery_stop(self):
        self.timer.signal_stop()

    def get_evaluation(self):
        return self.timer.get_total()
