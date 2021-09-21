from typing import Protocol
from active_learning_ts.oracle import Oracle
from active_learning_ts.query_selection.selection_criteria import SelectionCriteria


class QueryOptimizer(Protocol):
    def optimize_query_candidates(
        self, selection_criteria: SelectionCriteria, query_subject: Oracle
    ):
        pass
