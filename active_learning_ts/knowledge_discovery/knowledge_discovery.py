from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel


class KnowledgeDiscovery:
    def __init__(self, knowledge_discovery_task: KnowledgeDiscoveryTask, surrogate_model: SurrogateModel,
                 surrogate_sampler: QuerySampler, num_queries: int):
        self.knowledge_discovery_task = knowledge_discovery_task
        self.surrogate_model = surrogate_model
        self.sampler = surrogate_sampler
        self.num_queries = num_queries

    def discover(self):
        if self.num_queries == 0:
            return
        self.sampler.post_init(self.surrogate_model.get_query_pool())
        x = self.sampler.sample(num_queries=self.num_queries)
        x, y = self.surrogate_model.query(x)
        self.knowledge_discovery_task.learn(x, y)
