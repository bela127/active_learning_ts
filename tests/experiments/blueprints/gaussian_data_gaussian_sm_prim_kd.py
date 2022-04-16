from distribution_data_generation.data_sources.multi_gausian_data_source import MultiGausianDataSource

from active_learning_ts.data_retrievement.augmentation.no_augmentation import NoAugmentation
from active_learning_ts.data_retrievement.interpolation_strategies.flat_map_interpolation import \
    FlatMapInterpolation
from active_learning_ts.data_retrievement.retrievement_strategies.exact_retrievement import ExactRetrievement
from active_learning_ts.evaluation.evaluation_metrics.avg_round_time_evaluator import AvgRoundTimeEvaluator
from active_learning_ts.experiments.blueprint_element import BlueprintElement
from active_learning_ts.instance_properties.costs.constant_instance_cost import ConstantInstanceCost
from active_learning_ts.instance_properties.objectives.constant_instance_objective import ConstantInstanceObjective
from active_learning_ts.knowledge_discovery.discover_tasks.prim.prim_scenario_discovery_knowledge_discovery_task import \
    PrimScenarioDiscoveryKnowledgeDiscoveryTaskConfig
from active_learning_ts.query_selection.query_optimizers.max_entropy_query_optimizer import MaximumEntropyQueryOptimizer
from active_learning_ts.query_selection.query_samplers.random_query_sampler import RandomContinuousQuerySampler
from active_learning_ts.query_selection.selection_criterias.composite_selection_criteria import \
    CompositeSelectionCriteria
from active_learning_ts.query_selection.selection_criterias.explore_selection_criteria import ExploreSelectionCriteria
from active_learning_ts.query_selection.selection_criterias.knowledge_uncertainty_selection_criteria import \
    KnowledgeUncertaintySelectionCriteria
from active_learning_ts.surrogate_model.surrogate_models.gaussion_surrogate_model import GaussianSurrogateModel
from active_learning_ts.training.training_strategies.direct_training_strategy import DirectTrainingStrategy

repeat = 1

learning_steps = 10

data_source = BlueprintElement[MultiGausianDataSource]({'in_dim': 2, 'out_dim': 1, 'min_x': -5, 'max_x': 5})
retrievement_strategy = BlueprintElement[ExactRetrievement]()
interpolation_strategy = BlueprintElement[FlatMapInterpolation]()

augmentation_pipeline = BlueprintElement[NoAugmentation]()

instance_level_objective = BlueprintElement[ConstantInstanceObjective]()
instance_cost = BlueprintElement[ConstantInstanceCost]()

surrogate_model = BlueprintElement[GaussianSurrogateModel]()
training_strategy = BlueprintElement[DirectTrainingStrategy]()

# TODO: HMMMMM might be problematic, problem due to composite pattern
selection_criteria = BlueprintElement[CompositeSelectionCriteria](
    {'selection_criteria': [BlueprintElement[KnowledgeUncertaintySelectionCriteria](),
                            BlueprintElement[ExploreSelectionCriteria]()]})
surrogate_sampler = BlueprintElement[RandomContinuousQuerySampler]()
query_optimizer = BlueprintElement[MaximumEntropyQueryOptimizer]({'num_tries': 10})

num_knowledge_discovery_queries = 100
knowledge_discovery_sampler = BlueprintElement[RandomContinuousQuerySampler]()
knowledge_discovery_task = PrimScenarioDiscoveryKnowledgeDiscoveryTaskConfig(y_max=1.0, y_min=0.)

evaluation_metrics = [BlueprintElement[AvgRoundTimeEvaluator]()]
