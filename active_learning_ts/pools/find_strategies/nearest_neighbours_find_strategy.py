from typing import List

from active_learning_ts.pools.find_strategy import FindStrategy

import tensorflow as tf
from scipy.spatial import KDTree


class NearestNeighboursFindStrategy(FindStrategy):
    def __init__(self, num_neighbours: int):
        self.num_neighbours = num_neighbours
        self.kd_tree = None

    def post_init(self, data_set):
        self.data_set = data_set
        self.kd_tree = KDTree(data_set)

    def _find(self, point: tf.Tensor) -> List[tf.Tensor]:
        distances, nn = self.kd_tree.query(point, k=self.num_neighbours)

        out = []
        for i in nn:
            out.append(self.data_set[i])
        return out
