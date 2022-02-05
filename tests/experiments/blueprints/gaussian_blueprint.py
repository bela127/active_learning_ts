from distribution_data_generation.data_sources.multi_gausian_data_source import MultiGausianDataSource

from active_learning_ts.data_retrievement.augmentation.no_augmentation import NoAugmentation
from active_learning_ts.data_retrievement.interpolation.interpolation_strategies.flat_map_interpolation import \
    FlatMapInterpolation
from active_learning_ts.evaluation.evaluation_metrics.avg_round_time_evaluator import AvgRoundTimeEvaluator
from active_learning_ts.evaluation.evaluation_metrics.rounder_counter_evaluator import RoundCounterEvaluator
from active_learning_ts.experiments.blueprint import Blueprint
from active_learning_ts.instance_properties.costs.constant_instance_cost import ConstantInstanceCost
from active_learning_ts.instance_properties.objectives.constant_instance_objective import ConstantInstanceObjective
from active_learning_ts.knowledge_discovery.no_knowledge_discovery_task import NoKnowledgeDiscoveryTask
from active_learning_ts.pools.retrievement_strategies.exact_retrievement import ExactRetrievement
from active_learning_ts.query_selection.query_optimizers.fixed_value_optimizer import FixedValueOptimizer
from active_learning_ts.query_selection.query_samplers.random_query_sampler import RandomContinuousQuerySampler
from active_learning_ts.query_selection.selection_criterias.explore_selection_criteria import ExploreSelectionCriteria
from active_learning_ts.surrogate_models.gaussion_surrogate_model import GaussianSurrogateModel
from active_learning_ts.training.training_strategies.direct_training_strategy import DirectTrainingStrategy


class GaussianBlueprint(Blueprint):
    repeat = 1

    def __init__(self):
        self.learning_steps = 20
        self.num_knowledge_discovery_queries = 0

        # TODO, not having 0 between min and max causes a logic error. Investigate
        self.data_source = MultiGausianDataSource(in_dim=3, out_dim=2, min_x=-5, max_x=5)
        self.retrievement_strategy = ExactRetrievement()
        self.interpolation_strategy = FlatMapInterpolation()

        self.augmentation_pipeline = NoAugmentation()

        self.instance_level_objective = ConstantInstanceObjective()
        self.instance_cost = ConstantInstanceCost()

        self.surrogate_model = GaussianSurrogateModel()
        self.training_strategy = DirectTrainingStrategy()

        self.selection_criteria = ExploreSelectionCriteria()
        self.surrogate_sampler = RandomContinuousQuerySampler()
        self.query_optimizer = FixedValueOptimizer()

        self.knowledge_discovery_sampler = RandomContinuousQuerySampler()
        self.knowledge_discovery_task = NoKnowledgeDiscoveryTask()

        self.evaluation_metrics = [AvgRoundTimeEvaluator(), RoundCounterEvaluator()]
