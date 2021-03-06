from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric


class RoundCounterEvaluator(EvaluationMetric):
    """
    counts the number of rounds. Evaluation is the number of rounds
    """
    def __init__(self):
        self.round_number = 0

    def eval(self):
        self.round_number += 1

    def get_evaluation(self):
        return self.round_number
