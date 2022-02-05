import tensorflow as tf
from distribution_data_generation.data_sources.data_set_data_source import DataSetDataSource

from active_learning_ts.data_retrievement.augmentation.no_augmentation import NoAugmentation
from active_learning_ts.data_retrievement.interpolation.interpolation_strategies.flat_map_interpolation import \
    FlatMapInterpolation
from active_learning_ts.evaluation.evaluation_metrics.total_knowledge_discovery_time_evaluator import \
    TotalKnowledgeDiscoveryTimeEvaluator
from active_learning_ts.evaluation.evaluation_metrics.total_query_time_evaluator import TotalQueryTimeEvaluator
from active_learning_ts.evaluation.evaluation_metrics.total_training_time_evaluator import TotalTrainingTimeEvaluator
from active_learning_ts.experiments.blueprint import Blueprint
from active_learning_ts.instance_properties.costs.constant_instance_cost import ConstantInstanceCost
from active_learning_ts.instance_properties.objectives.constant_instance_objective import ConstantInstanceObjective
from active_learning_ts.knowledge_discovery.extreme_point.maxima_knowledge_task import MaximaKnowledgeDiscoveryTask
from active_learning_ts.pools.retrievement_strategies.nearest_neighbours_retreivement_strategy import \
    NearestNeighboursFindStrategy
from active_learning_ts.query_selection.query_optimizers.fixed_value_optimizer import FixedValueOptimizer
from active_learning_ts.query_selection.query_samplers.random_query_sampler import RandomContinuousQuerySampler
from active_learning_ts.query_selection.selection_criterias.explore_selection_criteria import ExploreSelectionCriteria
from active_learning_ts.surrogate_models.gaussion_surrogate_model import GaussianSurrogateModel
from active_learning_ts.training.training_strategies.direct_training_strategy import DirectTrainingStrategy


class MaximaKnowledgeDiscovery(Blueprint):
    repeat = 1

    def __init__(self):
        self.learning_steps = 10
        self.num_knowledge_discovery_queries = 1000

        x = tf.random.uniform((1000, 3), -10.0, 10.0)
        y = tf.reshape(tf.convert_to_tensor(tf.reduce_min(x, 1)), (1000, 1))
        y = 10 - (y * y)

        self.data_source = DataSetDataSource(x, y)
        self.retrievement_strategy = NearestNeighboursFindStrategy(5)
        self.interpolation_strategy = FlatMapInterpolation()

        self.augmentation_pipeline = NoAugmentation()

        self.instance_level_objective = ConstantInstanceObjective()
        self.instance_cost = ConstantInstanceCost()

        self.surrogate_model = GaussianSurrogateModel()
        self.training_strategy = DirectTrainingStrategy()

        self.selection_criteria = ExploreSelectionCriteria()
        self.surrogate_sampler = RandomContinuousQuerySampler()
        self.query_optimizer = FixedValueOptimizer(0.3)

        self.knowledge_discovery_sampler = RandomContinuousQuerySampler()
        self.knowledge_discovery_task = MaximaKnowledgeDiscoveryTask()

        self.evaluation_metrics = [TotalQueryTimeEvaluator(), TotalTrainingTimeEvaluator(),
                                   TotalKnowledgeDiscoveryTimeEvaluator()]
