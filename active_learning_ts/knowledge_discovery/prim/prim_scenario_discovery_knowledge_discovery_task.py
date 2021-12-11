from typing import List

import numpy as np

from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
import tensorflow as tf

from active_learning_ts.knowledge_discovery.prim.prim import PRIM


class PrimScenarioDiscoveryKnowledgeDiscoveryTask(KnowledgeDiscoveryTask):
    def __init__(self, num_queries: int = 1000, y_max: float = 1.0, y_min: float = 0.):
        self.num_queries = num_queries
        self.y_max = tf.convert_to_tensor(y_max)
        self.y_min = tf.convert_to_tensor(y_min)
        self.range = self.y_max - self.y_min
        self.prim = PRIM()
        self.boxes = []
        self.num_boxes = 0.

    def learn(self, data_set: tf.Tensor, data_values: tf.Tensor):
        x = data_set
        y = (data_values - self.y_min) / self.range

        self.prim.fit(x, y)

        self.boxes.append((tf.convert_to_tensor(self.prim.box_[0], dtype=np.float32),
                           tf.convert_to_tensor(self.prim.box_[1], dtype=np.float32)))
        self.num_boxes += 1.

    def _uncertainty(self, point: tf.Tensor) -> float:
        in_boxes = 0.
        for a, b in self.boxes:
            in_boxes += tf.case([(tf.reduce_any(tf.math.less(point, a)), lambda: 0.0),
                                 (tf.reduce_any(tf.math.greater(point, b)), lambda: 0.0)], default=lambda: 1.0)
        not_in_boxes = self.num_boxes - in_boxes
        return tf.cast(tf.math.abs(tf.math.abs(in_boxes - not_in_boxes) - self.num_boxes) / self.num_boxes,
                       dtype=tf.dtypes.float32)

    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        return tf.map_fn(lambda t: self._uncertainty(t), points, parallel_iterations=10)
