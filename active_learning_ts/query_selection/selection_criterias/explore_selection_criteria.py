from typing import List

import tensorflow as tf

from active_learning_ts.query_selection.selection_criteria import SelectionCriteria


class ExploreSelectionCriteria(SelectionCriteria):
    """
    Used to encourage the model to explore the search space as much as possible, by encouraging training in areas where
    uncertainty is high
    """

    def score_queries(self, queries: tf.Tensor) -> tf.Tensor:
        out = self.surrogate_model.uncertainty(queries)
        return tf.reshape(out, (out.shape[0], 1))
