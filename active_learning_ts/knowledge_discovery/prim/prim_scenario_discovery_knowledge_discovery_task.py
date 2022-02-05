import numpy as np
import tensorflow as tf

from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.knowledge_discovery.prim.prim import PRIM
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.queryable import Queryable


class PrimScenarioDiscoveryKnowledgeDiscoveryTask(KnowledgeDiscoveryTask):
    def __init__(self, y_max: float = 1.0, y_min: float = 0.):
        if y_max <= y_min:
            raise ValueError('The minimum value cannot be greater or equal to the maximum value.')
        y_max = float(y_max)
        y_min = float(y_min)
        self.y_max = tf.convert_to_tensor(y_max, dtype=tf.dtypes.float32)
        self.y_min = tf.convert_to_tensor(y_min, dtype=tf.dtypes.float32)
        self.range = self.y_max - self.y_min
        self.prim = PRIM()
        self.boxes = []
        self.num_boxes = 0.

    def post_init(self, surrogate_model: Queryable, sampler: QuerySampler):
        super(PrimScenarioDiscoveryKnowledgeDiscoveryTask, self).post_init(surrogate_model, sampler)
        if not (surrogate_model.point_shape == (2,) and surrogate_model.value_shape == (1,)):
            raise ValueError('PrimScenarioDiscoveryKnowledgeDiscoveryTask requires a vector Surrogate input dimension '
                             '2 and output dimension 1')

    def learn(self, num_queries):
        if num_queries == 0:
            return
        x = self.sampler.sample(num_queries=num_queries)
        data_set, data_values = self.surrogate_model.query(x)
        x = data_set
        y = (data_values - self.y_min) / self.range

        self.prim.fit(x, y)
        self.boxes.append((tf.convert_to_tensor(self.prim.box_[0], dtype=np.float32),
                           tf.convert_to_tensor(self.prim.box_[1], dtype=np.float32)))
        self.num_boxes += 1.

    def _uncertainty(self, point: tf.Tensor) -> float:
        if len(self.boxes) == 0:
            return tf.convert_to_tensor(.0, dtype=tf.dtypes.float32)
        in_boxes = 0.
        for a, b in self.boxes:
            in_boxes += tf.case([(tf.reduce_any(tf.math.less(point, a)), lambda: 0.0),
                                 (tf.reduce_any(tf.math.greater(point, b)), lambda: 0.0)], default=lambda: 1.0)
        not_in_boxes = self.num_boxes - in_boxes
        return tf.cast(tf.math.abs(tf.math.abs(in_boxes - not_in_boxes) - self.num_boxes) / self.num_boxes,
                       dtype=tf.dtypes.float32)

    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        return tf.map_fn(lambda t: self._uncertainty(t), points, parallel_iterations=10)
