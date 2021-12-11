from typing import List
import tensorflow as tf

from active_learning_ts.query_selection.selection_criteria import SelectionCriteria


class CompositeSelectionCriteria(SelectionCriteria):
    def __init__(self, selection_criteria: List[SelectionCriteria]):
        self.selection_criteria = selection_criteria

    def score_queries(self, queries: tf.Tensor) -> tf.Tensor:
        tensors = []

        for criteria in self.selection_criteria:
            tensors.append(criteria.score_queries(queries))

        return tf.concat(tensors, 1)
