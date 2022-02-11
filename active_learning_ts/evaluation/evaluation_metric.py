from typing import Protocol

from active_learning_ts.data_blackboard import Blackboard


class EvaluationMetric(Protocol):
    """
    Considering all parts of the experiment, (passed in as blueprint), measures certain metrics about the experiment,
    such as total number of elapsed rounds, average/total training time, uncertainty improvement per round...
    """

    def post_init(self, blackboard: Blackboard, blueprint):
        self.blackboard = blackboard
        self.blueprint = blueprint
        self.end_experiment = None

    def eval(self) -> None:
        """
        Evaluates the metric after the current round
        :return: None
        """
        pass

    def get_evaluation(self):
        """
        Returns all evaluations as a dictionary, or object when using aggregate metrics
        :return: all evaluations as an object or dictionary
        """
        pass

    def signal_learn_stop(self):
        pass

    def signal_learn_start(self):
        pass

    def signal_round_stop(self):
        pass

    def signal_round_start(self):
        pass

    def signal_query_stop(self):
        pass

    def signal_query_start(self):
        pass

    def signal_evaluation_start(self):
        pass

    def signal_evaluation_stop(self):
        pass

    def signal_knowledge_discovery_start(self):
        pass

    def signal_knowledge_discovery_stop(self):
        pass

    def signal_end_of_experiment(self):
        pass
