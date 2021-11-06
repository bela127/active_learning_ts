from typing import Protocol, List, Union
import tensorflow as tf

from active_learning_ts.pool import Pool
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel


class SelectionCriteria(Protocol):
    """
    Selection Criteria aims to score any given point, on how good of a query it would be. This is used in order to, with
    the specific task in mind, be able to select the best query of a bunch of different queries
    """

    def score_queries(self, queries: List[tf.Tensor]) -> tf.Tensor:
        """
        Scores the given queries on how good they are. The higher the number the better
        :param queries: the points to be scored
        :return: a tensor of their respective scores
        """
        pass

    def post_init(self, surrogate_model: SurrogateModel):
        self.surrogate_model = surrogate_model

