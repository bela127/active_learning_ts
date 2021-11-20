from typing import List

from active_learning_ts.data_retrievement.data_source import DataSource
from active_learning_ts.pools.retrievement_strategy import RetrievementStrategy

import tensorflow as tf
from scipy.spatial import KDTree


class NearestNeighboursFindStrategy(RetrievementStrategy):
    def __init__(self, num_neighbours: int):
        self.num_neighbours = num_neighbours
        self.kd_tree = None

    def post_init(self, data_source: DataSource):
        if not data_source.possible_queries().is_discrete():
            raise TypeError("Nearest Neighbour requires a discrete pool")
        self.data_set = data_source.possible_queries().get_all_elements()
        self.kd_tree = KDTree(self.data_set)

    def _find(self, point: tf.Tensor) -> List[tf.Tensor]:
        distances, nn = self.kd_tree.query(point, k=self.num_neighbours)

        out = []
        for i in nn:
            out.append(self.data_set[i])
        return out
