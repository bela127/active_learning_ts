import tensorflow as tf

from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask


class NoKnowledgeDiscoveryTask(KnowledgeDiscoveryTask):
    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        return tf.convert_to_tensor([.0] * len(points), dtype=tf.dtypes.float32)
