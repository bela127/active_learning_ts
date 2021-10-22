from typing import List

from active_learning_ts.data_blackboard import Blackboard
from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.experiments.blueprint import Blueprint


class Evaluator:
    # TODO: add some sort of signal method for time sensitive metrics. This will tell them when the model is learning,
    #   and when evaluations are taking place
    def __init__(self, evaluation_metrics: List[EvaluationMetric], blackboard: Blackboard, blueprint: Blueprint):
        self.evaluation_metrics = evaluation_metrics
        for em in evaluation_metrics:
            em.post_init(blackboard, blueprint)

    def evaluate(self):
        for em in self.evaluation_metrics:
            em.eval()

    def get_evaluations(self):
        return [x.get_evaluation() for x in self.evaluation_metrics]
