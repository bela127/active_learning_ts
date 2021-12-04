from active_learning_ts.evaluation.evaluator import Evaluator
from active_learning_ts.knowledge_discovery.knowledge_discovery import KnowledgeDiscovery
from active_learning_ts.training.trainer import Trainer
from active_learning_ts.oracle import Oracle
from active_learning_ts.query_selection.query_selector import QuerySelector
from active_learning_ts.data_blackboard import Blackboard


class ActiveLearner:
    """
    Responsible for executing each learning step.

    A learning step includes query selection, Data retrievement, model training, and evaluation
    """
    def __init__(
        self,
        oracle: Oracle,
        query_selector: QuerySelector,
        blackboard: Blackboard,
        trainer: Trainer,
        knowledge_discovery: KnowledgeDiscovery,
        evaluator: Evaluator
    ) -> None:
        self.query_selector: QuerySelector = query_selector
        self.oracle: Oracle = oracle
        self.blackboard: Blackboard = blackboard
        self.trainer: Trainer = trainer
        self.knowledge_discovery = knowledge_discovery
        self.evaluator = evaluator

    def learning_step(self):
        self.evaluator.signal_round_start()

        self.evaluator.signal_query_start()
        query_candidates = self.query_selector.select()
        self.oracle.query(query_candidates)
        self.evaluator.signal_query_stop()

        self.evaluator.signal_learn_start()
        self.trainer.train()
        self.evaluator.signal_learn_stop()

        self.knowledge_discovery.discover()

        self.evaluator.signal_round_stop()

        self.evaluator.evaluate()
