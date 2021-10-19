from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
from active_learning_ts.query_selection.query_optimizer import QueryOptimizer
from active_learning_ts.oracle import Oracle
from active_learning_ts.data_blackboard import Blackboard


class QuerySelector:
    def __init__(
        self,
        blackboard: Blackboard,
        query_optimizer: QueryOptimizer,
        selection_criteria: SelectionCriteria,
        query_subject: Oracle,
    ) -> None:
        self.blackboard: Blackboard = blackboard
        self.query_optimizer = query_optimizer
        self.selection_criteria = selection_criteria
        self.query_subject = query_subject

    def select(self):
        query_candidates = self.query_optimizer.optimize_query_candidates()
        return query_candidates
