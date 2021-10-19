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
        query_candidates = self.query_optimizer.optimize_query_candidates(
            # TODO: the only reason one would pass the selection criteria and the query subject is if they change
            # midway through the experiment
            # self.selection_criteria, self.query_subject

            #True, they can be passed in the init
        )
        return query_candidates
