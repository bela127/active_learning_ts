from typing import List

from active_learning_ts.data_blackboard import Blackboard
from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.experiments.blueprint_instance import BlueprintInstance


class Evaluator:
    """
    Holds a list of the evaluation metrics, calls them to evaluate after every round
    """

    def __init__(self, evaluation_metrics: List[EvaluationMetric], blackboard: Blackboard, blueprint: BlueprintInstance):
        self.evaluation_metrics = evaluation_metrics
        self.__do_learn = True
        self.__do_query = True
        self.__do_discover_knowledge = True
        self.__do_evaluate = True
        self.__experiment_end = False
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
        out = []
        f = lambda x: '[' + ', '.join([str(a) for a in x]) + ']' if isinstance(x, list) else str(x)
        [out.append('"' + type(x).__name__ + '" : ' + f(x.get_evaluation())) for x in self.evaluation_metrics]
        return '{\n' + ',\n'.join(out) + '\n}\n'

    def get_do_learn(self):
        return self.__do_learn

    def get_do_query(self):
        return self.__do_query

    def get_do_discover_knowledge(self):
        return self.__do_discover_knowledge

    def get_do_evaluate(self):
        return self.__do_evaluate

    def _set_do_learn(self, do_learn):
        self.__do_learn = do_learn

    def _set_do_query(self, do_query):
        self.__do_query = do_query

    def _set_do_discover_knowledge(self, do_discover_knowledge):
        self.__do_discover_knowledge = do_discover_knowledge

    def _set_do_evaluate(self, do_evaluate):
        self.__do_evaluate = do_evaluate

    def signal_end_of_experiment(self):
        [e.signal_end_of_experiment() for e in self.evaluation_metrics]

    def get_end_of_experiment(self) -> bool:
        f = lambda x, y: y if x is None else (x and y if y is not None else x)
        out = None

        for e in self.evaluation_metrics:
            out = f(out, e.end_experiment)
            # Do not simplify this. This is not equivalent to 'not out' in the case that out is None
            if out is False:
                break
        return out if out is not None else False
