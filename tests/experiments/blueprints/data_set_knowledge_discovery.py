import tensorflow as tf
from distribution_data_generation.data_sources.data_set_data_source import DataSetDataSource

from active_learning_ts.data_retrievement.augmentation.no_augmentation import NoAugmentation
from active_learning_ts.data_retrievement.interpolation.interpolation_strategies.flat_map_interpolation import \
    FlatMapInterpolation
from active_learning_ts.evaluation.evaluation_metrics.rounder_counter_evaluator import RoundCounterEvaluator
from active_learning_ts.experiments.blueprint import Blueprint
from active_learning_ts.instance_properties.costs.constant_instance_cost import ConstantInstanceCost
from active_learning_ts.instance_properties.objectives.constant_instance_objective import ConstantInstanceObjective
from active_learning_ts.knowledge_discovery.prim.prim_scenario_discovery_knowledge_discovery_task import \
    PrimScenarioDiscoveryKnowledgeDiscoveryTask
from active_learning_ts.pools.retrievement_strategies.exact_retrievement import ExactRetrievement
from active_learning_ts.query_selection.query_optimizers.no_query_optimizer import NoQueryOptimizer
from active_learning_ts.query_selection.query_samplers.random_query_sampler import RandomContinuousQuerySampler
from active_learning_ts.query_selection.selection_criterias.no_selection_criteria import NoSelectionCriteria
from active_learning_ts.surrogate_models.no_surrogate_model import NoSurrogateModel
from active_learning_ts.training.training_strategies.no_training_strategy import NoTrainingStrategy

x = tf.random.uniform(shape=(1000, 2)) * 100
y = [tf.constant([0.9]) if 50 <= a[0] <= 60 and 10 <= a[1] <= 20 else tf.constant([0.0]) for a in x]


class DataSetKnowledgeDiscovery(Blueprint):
    repeat = 1

    def __init__(self):
        self.learning_steps = 1
        self.num_knowledge_discovery_queries = 100

        self.data_source = DataSetDataSource(data_points=x, data_values=tf.convert_to_tensor(y))
        self.retrievement_strategy = ExactRetrievement()
        self.interpolation_strategy = FlatMapInterpolation()

        self.augmentation_pipeline = NoAugmentation()

        self.instance_level_objective = ConstantInstanceObjective()
        self.instance_cost = ConstantInstanceCost()

        self.surrogate_model = NoSurrogateModel()
        self.training_strategy = NoTrainingStrategy()

        self.selection_criteria = NoSelectionCriteria()
        self.surrogate_sampler = RandomContinuousQuerySampler()
        self.query_optimizer = NoQueryOptimizer()

        self.knowledge_discovery_sampler = RandomContinuousQuerySampler()
        self.knowledge_discovery_task = PrimScenarioDiscoveryKnowledgeDiscoveryTask()

        self.evaluation_metrics = [RoundCounterEvaluator()]
