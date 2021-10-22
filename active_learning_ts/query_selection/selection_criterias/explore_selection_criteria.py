from typing import List

import tensorflow as tf

from active_learning_ts.query_selection.selection_criteria import SelectionCriteria


class ExploreSelectionCriteria(SelectionCriteria):

    def score_queries(self, queries: List[tf.Tensor]) -> tf.Tensor:
        return self.surrogate_model.uncertainty(queries)
