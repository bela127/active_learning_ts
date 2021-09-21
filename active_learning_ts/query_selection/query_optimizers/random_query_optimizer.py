from active_learning_ts.oracle import Oracle
from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
from active_learning_ts.query_selection.query_optimizer import QueryOptimizer


class RandomQueryOptimizer(QueryOptimizer):
    def optimize_query_candidates(
        self, selection_criteria: SelectionCriteria, query_subject: Oracle
    ):
        optimized_query_candidates = ["1"] #TODO
        return optimized_query_candidates