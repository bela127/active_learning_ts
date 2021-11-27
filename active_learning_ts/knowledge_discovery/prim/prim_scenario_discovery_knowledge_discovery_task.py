from typing import List

import numpy as np

from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
import tensorflow as tf

from active_learning_ts.knowledge_discovery.prim.prim import PRIM


class PrimScenarioDiscoveryKnowledgeDiscoveryTask(KnowledgeDiscoveryTask):
    def __init__(self, num_queries: int = 1000, y_max: float = 1.0):
        self.num_queries = num_queries
        self.y_max = y_max
        self.prim = PRIM()
        self.boxes = []
        self.num_boxes = 0.

    def learn(self):
        x = tf.random.uniform(shape=(self.num_queries, 2))
        x = tf.convert_to_tensor(self.surrogate_pool.get_elements_normalized(x))
        x = tf.reshape(x, [self.num_queries, x.shape[1] * x.shape[2]])
        y = tf.convert_to_tensor(self.surrogate_model.query(x)) / self.y_max

        self.prim.fit(x, y)

        self.boxes.append((tf.convert_to_tensor(self.prim.box_[0], dtype=np.float32),
                           tf.convert_to_tensor(self.prim.box_[1], dtype=np.float32)))
        self.num_boxes += 1.

    def _uncertainty(self, point: tf.Tensor) -> float:
        in_boxes = 0.
        for a, b in self.boxes:
            in_boxes += tf.case([(tf.reduce_any(tf.math.less(point, a)), lambda: 0),
                                 (tf.reduce_any(tf.math.greater(point, b)), lambda: 0)], default=lambda: 1)
        not_in_boxes = self.num_boxes - in_boxes
        return tf.math.abs(tf.math.abs(in_boxes - not_in_boxes) - self.num_boxes) / self.num_boxes

    def uncertainty(self, points: List[tf.Tensor]) -> tf.Tensor:
        return tf.map_fn(lambda t: self._uncertainty(t), points, parallel_iterations=10)
