from typing import List

from active_learning_ts.data_blackboard import Blackboard
from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.experiments.blueprint import Blueprint


class Evaluator:
    """
    Holds a list of the evaluation metrics, calls them to evaluate after every round
    """

    # TODO: add some sort of signal method for time sensitive metrics. This will tell them when the model is learning,
    #   and when evaluations are taking place
    def __init__(self, evaluation_metrics: List[EvaluationMetric], blackboard: Blackboard, blueprint: Blueprint):
        self.evaluation_metrics = evaluation_metrics
        for em in evaluation_metrics:
            em.post_init(blackboard, blueprint)

    def signal_learn_stop(self):
        for em in self.evaluation_metrics:
            em.signal_learn_stop()

    def signal_learn_start(self):
        for em in self.evaluation_metrics:
            em.signal_learn_start()

    def signal_round_stop(self):
        for em in self.evaluation_metrics:
            em.signal_round_stop()

    def signal_round_start(self):
        for em in self.evaluation_metrics:
            em.signal_round_start()

    def signal_query_stop(self):
        for em in self.evaluation_metrics:
            em.signal_query_stop()

    def signal_query_start(self):
        for em in self.evaluation_metrics:
            em.signal_query_start()

    def signal_evaluation_start(self):
        for em in self.evaluation_metrics:
            em.signal_evaluation_start()

    def signal_evaluation_stop(self):
        for em in self.evaluation_metrics:
            em.signal_evaluation_stop()

    def signal_knowledge_discovery_start(self):
        for em in self.evaluation_metrics:
            em.signal_knowledge_discovery_start()

    def signal_knowledge_discovery_stop(self):
        for em in self.evaluation_metrics:
            em.signal_knowledge_discovery_stop()

    def evaluate(self) -> None:
        """
        Evaluates all evaluation metrics
        :return: None
        """
        for em in self.evaluation_metrics:
            em.eval()

    def get_evaluations(self):
        """
        returns a list of the evaluations of all evaluation metrics
        :return: list of the evaluations of all evaluation metrics
        """
        return [x.get_evaluation() for x in self.evaluation_metrics]
