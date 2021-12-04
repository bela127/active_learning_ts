from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel
import tensorflow as tf


class KnowledgeDiscovery:
    def __init__(self, knowledge_discovery_task: KnowledgeDiscoveryTask, surrogate_model: SurrogateModel,
                 surrogate_sampler: QuerySampler, num_queries : int):
        self.knowledge_discovery_task = knowledge_discovery_task
        self.surrogate_model = surrogate_model
        self.sampler = surrogate_sampler
        self.num_queries = num_queries

    def discover(self):
        if self.num_queries == 0:
            return
        x = tf.convert_to_tensor(self.sampler.sample(num_queries=self.num_queries))
        y = tf.convert_to_tensor(self.surrogate_model.query(x))
        self.knowledge_discovery_task.learn(x, y)
