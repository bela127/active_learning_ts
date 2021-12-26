from active_learning_ts.query_selection.query_optimizer import QueryOptimizer


class NoQueryOptimizer(QueryOptimizer):
    def optimize_query_candidates(
            self
    ):
        return []
