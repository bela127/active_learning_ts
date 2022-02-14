from active_learning_ts.data_retrievement.augmentation.no_augmentation import (
    NoAugmentation,
)
from active_learning_ts.data_retrievement.data_sources.test_data_source import (
    TestDataSource,
)
from active_learning_ts.data_retrievement.interpolation_strategies.flat_map_interpolation import \
    FlatMapInterpolation
from active_learning_ts.evaluation.evaluation_metrics.rounder_counter_evaluator import RoundCounterEvaluator
from active_learning_ts.experiments.blueprint_element import BlueprintElement
from active_learning_ts.instance_properties.costs.constant_instance_cost import (
    ConstantInstanceCost,
)
from active_learning_ts.instance_properties.objectives.constant_instance_objective import (
    ConstantInstanceObjective,
)
from active_learning_ts.knowledge_discovery.discover_tasks.no_knowledge_discovery_task import NoKnowledgeDiscoveryTask
from active_learning_ts.data_retrievement.retrievement_strategies.exact_retrievement import ExactRetrievement
from active_learning_ts.query_selection.query_optimizers.no_query_optimizer import NoQueryOptimizer
from active_learning_ts.query_selection.query_samplers.random_query_sampler import RandomContinuousQuerySampler
from active_learning_ts.query_selection.selection_criterias.no_selection_criteria import (
    NoSelectionCriteria,
)
from active_learning_ts.surrogate_model.surrogate_models.no_surrogate_model import NoSurrogateModel
from active_learning_ts.training.training_strategies.no_training_strategy import NoTrainingStrategy

repeat = 2

learning_steps = 10
num_knowledge_discovery_queries = 0

data_source = BlueprintElement[TestDataSource]()
retrievement_strategy = BlueprintElement[ExactRetrievement]()
augmentation_pipeline = BlueprintElement[NoAugmentation]()
interpolation_strategy = BlueprintElement[FlatMapInterpolation]()

instance_level_objective = BlueprintElement[ConstantInstanceObjective]()
instance_cost = BlueprintElement[ConstantInstanceCost]()

surrogate_model = BlueprintElement[NoSurrogateModel]()
training_strategy = BlueprintElement[NoTrainingStrategy]()

surrogate_sampler = BlueprintElement[RandomContinuousQuerySampler]()
query_optimizer = BlueprintElement[NoQueryOptimizer]()
selection_criteria = BlueprintElement[NoSelectionCriteria]()

knowledge_discovery_sampler = BlueprintElement[RandomContinuousQuerySampler]()
knowledge_discovery_task = BlueprintElement[NoKnowledgeDiscoveryTask]()

evaluation_metrics = [BlueprintElement[RoundCounterEvaluator]()]
