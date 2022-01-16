from distribution_data_generation.data_sources.multi_gausian_data_source import MultiGausianDataSource

from active_learning_ts.data_retrievement.augmentation.no_augmentation import NoAugmentation
from active_learning_ts.data_retrievement.interpolation.interpolation_strategies.flat_map_interpolation import \
    FlatMapInterpolation
from active_learning_ts.evaluation.evaluation_metrics.avg_round_time_evaluator import AvgRoundTimeEvaluator
from active_learning_ts.instance_properties.costs.constant_instance_cost import ConstantInstanceCost
from active_learning_ts.instance_properties.objectives.constant_instance_objective import ConstantInstanceObjective
from active_learning_ts.knowledge_discovery.prim.prim_scenario_discovery_knowledge_discovery_task import \
    PrimScenarioDiscoveryKnowledgeDiscoveryTask
from active_learning_ts.pools.retrievement_strategies.exact_retrievement import ExactRetrievement
from active_learning_ts.query_selection.query_optimizers.max_entropy_query_optimizer import MaximumEntropyQueryOptimizer
from active_learning_ts.query_selection.query_samplers.random_query_sampler import RandomContinuousQuerySampler
from active_learning_ts.query_selection.selection_criterias.composite_selection_criteria import \
    CompositeSelectionCriteria
from active_learning_ts.query_selection.selection_criterias.explore_selection_criteria import ExploreSelectionCriteria
from active_learning_ts.query_selection.selection_criterias.knowledge_uncertainty_selection_criteria import \
    KnowledgeUncertaintySelectionCriteria
from active_learning_ts.surrogate_models.gaussion_surrogate_model import GaussianSurrogateModel
from active_learning_ts.training.training_strategies.direct_training_strategy import DirectTrainingStrategy

repeat = 2
learning_steps = 10
num_knowledge_discovery_queries = 100

data_source = MultiGausianDataSource(in_dim=2, out_dim=1, min_x=-5, max_x=5)
retrievement_strategy = ExactRetrievement()
interpolation_strategy = FlatMapInterpolation()

augmentation_pipeline = NoAugmentation()

instance_level_objective = ConstantInstanceObjective()
instance_cost = ConstantInstanceCost()

surrogate_model = GaussianSurrogateModel()
training_strategy = DirectTrainingStrategy()

selection_criteria = CompositeSelectionCriteria([KnowledgeUncertaintySelectionCriteria(), ExploreSelectionCriteria()])
surrogate_sampler = RandomContinuousQuerySampler()
query_optimizer = MaximumEntropyQueryOptimizer(num_tries=10)

knowledge_discovery_sampler = RandomContinuousQuerySampler()
knowledge_discovery_task = PrimScenarioDiscoveryKnowledgeDiscoveryTask()

evaluation_metrics = [AvgRoundTimeEvaluator()]
