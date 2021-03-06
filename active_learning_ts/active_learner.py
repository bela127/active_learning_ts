from active_learning_ts.logging.data_blackboard import Blackboard
from active_learning_ts.evaluation.evaluator import Evaluator
from active_learning_ts.knowledge_discovery.knowledge_discovery import KnowledgeDiscovery
from active_learning_ts.oracle import Oracle
from active_learning_ts.query_selection.query_selector import QuerySelector
from active_learning_ts.training.trainer import Trainer


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

        if self.evaluator.get_do_query():
            self.evaluator.signal_query_start()
            query_candidates = self.query_selector.select()
            self.oracle.query(query_candidates)
            self.evaluator.signal_query_stop()

        if self.evaluator.get_do_learn():
            self.evaluator.signal_learn_start()
            self.trainer.train()
            self.evaluator.signal_learn_stop()

        if self.evaluator.get_do_discover_knowledge():
            self.evaluator.signal_knowledge_discovery_start()
            self.knowledge_discovery.discover()
            self.evaluator.signal_knowledge_discovery_stop()

        self.evaluator.signal_round_stop()

        if self.evaluator.get_do_evaluate():
            self.evaluator.signal_evaluation_start()
            self.evaluator.evaluate()
            self.evaluator.signal_evaluation_stop()
