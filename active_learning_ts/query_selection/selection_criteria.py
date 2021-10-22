from typing import Protocol, List
import tensorflow as tf

from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel


class SelectionCriteria(Protocol):

    def score_queries(self, queries: List[tf.Tensor]) -> tf.Tensor:
        pass

    def post_init(self, surrogate_model: SurrogateModel):
        self.surrogate_model = surrogate_model
