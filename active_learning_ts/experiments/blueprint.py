from typing import List

from active_learning_ts.data_pipeline import DataPipeline
from active_learning_ts.data_retrievement.data_source import DataSource
from active_learning_ts.data_retrievement.interpolation.interpolation_strategy import InterpolationStrategy
from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.experiments.blueprint_element import BlueprintElement
from active_learning_ts.instance_properties.instance_cost import InstanceCost
from active_learning_ts.instance_properties.instance_objective import InstanceObjective
from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.pools.retrievement_strategy import RetrievementStrategy
from active_learning_ts.query_selection.query_optimizer import QueryOptimizer
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
from active_learning_ts.surrogate_models.surrogate_model import SurrogateModel
from active_learning_ts.training.training_strategy import TrainingStrategy


class Blueprint:
    """
    A blueprint is created in order to set up an experiment.

    Following field MUST be in the blueprint file, with the same names
    """
    repeat: int
    learning_steps: int
    num_knowledge_discovery_queries: int
    data_source: BlueprintElement[DataSource] = None
    retrievement_strategy: BlueprintElement[RetrievementStrategy] = None
    augmentation_pipeline: BlueprintElement[DataPipeline] = None
    interpolation_strategy: BlueprintElement[InterpolationStrategy] = None

    instance_level_objective: BlueprintElement[InstanceObjective] = None
    instance_cost: BlueprintElement[InstanceCost] = None

    surrogate_model: BlueprintElement[SurrogateModel] = None
    training_strategy: BlueprintElement[TrainingStrategy] = None

    surrogate_sampler: BlueprintElement[QuerySampler] = None
    query_optimizer: BlueprintElement[QueryOptimizer] = None
    selection_criteria: BlueprintElement[SelectionCriteria] = None

    evaluation_metrics: List[BlueprintElement[EvaluationMetric]] = None

    knowledge_discovery_sampler: BlueprintElement[QuerySampler] = None
    knowledge_discovery_task: BlueprintElement[KnowledgeDiscoveryTask] = None
