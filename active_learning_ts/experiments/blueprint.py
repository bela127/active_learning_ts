from active_learning_ts.data_retrievement.interpolation.interpolation_strategy import InterpolationStrategy
from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.instance_properties.instance_objective import InstanceObjective
from active_learning_ts.instance_properties.instance_cost import InstanceCost
from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.pools.retrievement_strategy import RetrievementStrategy
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
from active_learning_ts.query_selection.query_optimizer import QueryOptimizer
from active_learning_ts.data_pipeline import DataPipeline
from active_learning_ts.data_retrievement.data_source import DataSource

from typing import Protocol, List

from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel
from active_learning_ts.training.training_strategy import TrainingStrategy


class Blueprint(Protocol):
    """
    A blueprint is created in order to set up an experiment.

    Following field MUST be in the blueprint file, with the same names
    """
    repeat: int
    learning_steps: int

    num_knowledge_discovery_queries: int

    data_source: DataSource
    retrievement_strategy: RetrievementStrategy
    augmentation_pipeline: DataPipeline
    interpolation_strategy: InterpolationStrategy

    instance_level_objective: InstanceObjective
    instance_cost: InstanceCost

    surrogate_model: SurrogateModel
    training_strategy: TrainingStrategy

    surrogate_sampler: QuerySampler
    query_optimizer: QueryOptimizer
    selection_criteria: SelectionCriteria

    evaluation_metrics: List[EvaluationMetric]

    knowledge_discovery_sampler: QuerySampler
    knowledge_discovery_task: KnowledgeDiscoveryTask
