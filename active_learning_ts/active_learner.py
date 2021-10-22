from active_learning_ts.evaluation.evaluator import Evaluator
from active_learning_ts.training.trainer import Trainer
from active_learning_ts.oracle import Oracle
from active_learning_ts.query_selection.query_selector import QuerySelector
from active_learning_ts.data_blackboard import Blackboard


class ActiveLearner:
    def __init__(
        self,
        oracle: Oracle,
        query_selector: QuerySelector,
        blackboard: Blackboard,
        trainer: Trainer,
        evaluator: Evaluator
    ) -> None:
        self.query_selector: QuerySelector = query_selector
        self.oracle: Oracle = oracle
        self.blackboard: Blackboard = blackboard
        self.trainer: Trainer = trainer
        self.evaluator = evaluator

    def learning_step(self):
        query_candidates = self.query_selector.select()

        self.oracle.query(query_candidates)
        self.trainer.train()
        self.evaluator.evaluate()
