from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
from active_learning_ts.query_selection.query_optimizer import QueryOptimizer
from active_learning_ts.oracle import Oracle
from active_learning_ts.logging.data_blackboard import Blackboard


class QuerySelector:
    """
    Uses the given query optimizer and selection Criteria in order to generate queries
    """
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
        """
        Generates queries
        :return: a list of generated query indices
        """
        query_candidates = self.query_optimizer.optimize_query_candidates()
        return query_candidates
