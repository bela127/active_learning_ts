from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.queryable import Queryable


class KnowledgeDiscovery:
    def __init__(self, knowledge_discovery_task: KnowledgeDiscoveryTask, surrogate_model: Queryable,
                 surrogate_sampler: QuerySampler, num_queries: int):
        self.knowledge_discovery_task = knowledge_discovery_task
        self.surrogate_model = surrogate_model
        self.sampler = surrogate_sampler
        self.num_queries = num_queries
        self.sampler.post_init(self.surrogate_model.get_query_pool())
        self.knowledge_discovery_task.post_init(surrogate_model, self.sampler)

    def discover(self):
        self.knowledge_discovery_task.learn(self.num_queries)
