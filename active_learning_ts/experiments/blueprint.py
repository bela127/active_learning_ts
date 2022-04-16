from typing import Iterable, Protocol

from active_learning_ts.data_pipeline import DataPipeline
from active_learning_ts.data_retrievement.augmentation.no_augmentation import NoAugmentation
from active_learning_ts.data_retrievement.data_source import DataSource
from active_learning_ts.data_retrievement.data_sources.test_data_source import TestDataSource
from active_learning_ts.data_retrievement.interpolation_strategies.flat_map_interpolation import FlatMapInterpolation
from active_learning_ts.data_retrievement.interpolation_strategy import InterpolationStrategy
from active_learning_ts.data_retrievement.retrievement_strategies.exact_retrievement import ExactRetrievement
from active_learning_ts.evaluation.evaluation_metric import EvaluationMetric
from active_learning_ts.experiments.blueprint_element import BlueprintElement
from active_learning_ts.instance_properties.costs.constant_instance_cost import ConstantInstanceCost
from active_learning_ts.instance_properties.instance_cost import InstanceCost
from active_learning_ts.instance_properties.instance_objective import InstanceObjective
from active_learning_ts.instance_properties.objectives.constant_instance_objective import ConstantInstanceObjective
from active_learning_ts.knowledge_discovery.discover_tasks.no_knowledge_discovery_task import NoKnowledgeDiscoveryTask
from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.data_retrievement.retrievement_strategy import RetrievementStrategy
from active_learning_ts.query_selection.query_optimizer import QueryOptimizer
from active_learning_ts.query_selection.query_optimizers.no_query_optimizer import NoQueryOptimizer
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.query_selection.query_samplers.no_query_sampler import NoQuerySampler
from active_learning_ts.query_selection.selection_criteria import SelectionCriteria
from active_learning_ts.query_selection.selection_criterias.no_selection_criteria import NoSelectionCriteria
from active_learning_ts.surrogate_model.surrogate_model import SurrogateModel
from active_learning_ts.surrogate_model.surrogate_models.no_surrogate_model import NoSurrogateModel
from active_learning_ts.training.training_strategies.no_training_strategy import NoTrainingStrategy
from active_learning_ts.training.training_strategy import TrainingStrategy


class Blueprint(Protocol):
    """
    A blueprint is created in order to set up an experiment.
    The config objects are used to instantiate experiment modules
    """
    repeat: int
    learning_steps: int
    num_knowledge_discovery_queries: int
    data_source: BlueprintElement[DataSource] = BlueprintElement[TestDataSource]()
    retrievement_strategy: BlueprintElement[RetrievementStrategy] = BlueprintElement[ExactRetrievement]()
    augmentation_pipeline: BlueprintElement[DataPipeline] = BlueprintElement[NoAugmentation]()
    interpolation_strategy: BlueprintElement[InterpolationStrategy] = BlueprintElement[FlatMapInterpolation]()

    instance_level_objective: BlueprintElement[InstanceObjective] = BlueprintElement[ConstantInstanceObjective]()
    instance_cost: BlueprintElement[InstanceCost] = BlueprintElement[ConstantInstanceCost]()

    surrogate_model: BlueprintElement[SurrogateModel] = BlueprintElement[NoSurrogateModel]()
    training_strategy: BlueprintElement[TrainingStrategy] = BlueprintElement[NoTrainingStrategy]()

    surrogate_sampler: BlueprintElement[QuerySampler] = BlueprintElement[NoQuerySampler]()
    query_optimizer: BlueprintElement[QueryOptimizer] = BlueprintElement[NoQueryOptimizer]()
    selection_criteria: BlueprintElement[SelectionCriteria] = BlueprintElement[NoSelectionCriteria]()

    evaluation_metrics: Iterable[BlueprintElement[EvaluationMetric]] = []

    knowledge_discovery_sampler: BlueprintElement[QuerySampler] = BlueprintElement[NoQuerySampler]()
    knowledge_discovery_task: BlueprintElement[KnowledgeDiscoveryTask] = BlueprintElement[NoKnowledgeDiscoveryTask]()
