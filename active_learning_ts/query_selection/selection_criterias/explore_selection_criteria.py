from typing import List

import tensorflow as tf

from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel


class ExploreSelectionCriteria(SelectionCriteria):
    def __init__(self, surrogate_model: SurrogateModel):
        self.surrogate_model = surrogate_model

    def score_queries(self, queries: List[tf.Tensor]) -> tf.Tensor:
        return self.surrogate_model.uncertainty(queries)
