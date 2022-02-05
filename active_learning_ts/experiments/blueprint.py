from typing import Protocol, List

from active_learning_ts.data_pipeline import DataPipeline
from active_learning_ts.data_retrievement.data_source import DataSource
from active_learning_ts.data_retrievement.interpolation.interpolation_strategy import InterpolationStrategy
from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.instance_properties.instance_cost import InstanceCost
from active_learning_ts.instance_properties.instance_objective import InstanceObjective
from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.pools.retrievement_strategy import RetrievementStrategy
from active_learning_ts.query_selection.query_optimizer import QueryOptimizer
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel
from active_learning_ts.training.training_strategy import TrainingStrategy


class Blueprint(Protocol):
    """
    A blueprint is created in order to set up an experiment.

    Following field MUST be in the blueprint file, with the same names
    """
    repeat: int

    def __init__(self):
        self.learning_steps: int

        self.num_knowledge_discovery_queries: int

        self.data_source: DataSource
        self.retrievement_strategy: RetrievementStrategy
        self.augmentation_pipeline: DataPipeline
        self.interpolation_strategy: InterpolationStrategy

        self.instance_level_objective: InstanceObjective
        self.instance_cost: InstanceCost

        self.surrogate_model: SurrogateModel
        self.training_strategy: TrainingStrategy

        self.surrogate_sampler: QuerySampler
        self.query_optimizer: QueryOptimizer
        self.selection_criteria: SelectionCriteria

        self.evaluation_metrics: List[EvaluationMetric]

        self.knowledge_discovery_sampler: QuerySampler
        self.knowledge_discovery_task: KnowledgeDiscoveryTask
