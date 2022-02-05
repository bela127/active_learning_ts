from active_learning_ts.data_retrievement.augmentation.no_augmentation import (
    NoAugmentation,
)
from active_learning_ts.data_retrievement.data_sources.test_data_source import (
    TestDataSource,
)
from active_learning_ts.data_retrievement.interpolation.interpolation_strategies.flat_map_interpolation import \
    FlatMapInterpolation
from active_learning_ts.evaluation.evaluation_metrics.rounder_counter_evaluator import RoundCounterEvaluator
from active_learning_ts.experiments.blueprint import Blueprint
from active_learning_ts.instance_properties.costs.constant_instance_cost import (
    ConstantInstanceCost,
)
from active_learning_ts.instance_properties.objectives.constant_instance_objective import (
    ConstantInstanceObjective,
)
from active_learning_ts.knowledge_discovery.no_knowledge_discovery_task import NoKnowledgeDiscoveryTask
from active_learning_ts.pools.retrievement_strategies.exact_retrievement import ExactRetrievement
from active_learning_ts.query_selection.query_optimizers.no_query_optimizer import NoQueryOptimizer
from active_learning_ts.query_selection.query_samplers.random_query_sampler import RandomContinuousQuerySampler
from active_learning_ts.query_selection.selection_criterias.no_selection_criteria import (
    NoSelectionCriteria,
)
from active_learning_ts.surrogate_models.no_surrogate_model import NoSurrogateModel
from active_learning_ts.training.training_strategies.no_training_strategy import NoTrainingStrategy


class TestBluePrint(Blueprint):
    repeat = 2

    def __init__(self):
        self.learning_steps = 10
        self.num_knowledge_discovery_queries = 0

        self.data_source = TestDataSource()
        self.retrievement_strategy = ExactRetrievement()
        self.augmentation_pipeline = NoAugmentation()
        self.interpolation_strategy = FlatMapInterpolation()

        self.instance_level_objective = ConstantInstanceObjective()
        self.instance_cost = ConstantInstanceCost()

        self.surrogate_model = NoSurrogateModel()
        self.training_strategy = NoTrainingStrategy()

        self.surrogate_sampler = RandomContinuousQuerySampler()
        self.query_optimizer = NoQueryOptimizer()
        self.selection_criteria = NoSelectionCriteria()

        self.knowledge_discovery_sampler = RandomContinuousQuerySampler()
        self.knowledge_discovery_task = NoKnowledgeDiscoveryTask()

        self.evaluation_metrics = [RoundCounterEvaluator()]
