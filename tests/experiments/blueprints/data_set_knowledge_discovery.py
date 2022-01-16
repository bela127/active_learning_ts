import tensorflow as tf
from distribution_data_generation.data_sources.data_set_data_source import DataSetDataSource

from active_learning_ts.data_retrievement.augmentation.no_augmentation import NoAugmentation
from active_learning_ts.data_retrievement.interpolation.interpolation_strategies.flat_map_interpolation import \
    FlatMapInterpolation
from active_learning_ts.evaluation.evaluation_metrics.rounder_counter_evaluator import RoundCounterEvaluator
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

repeat = 2
learning_steps = 1
num_knowledge_discovery_queries = 100

data_source = DataSetDataSource(data_points=x, data_values=tf.convert_to_tensor(y))
retrievement_strategy = ExactRetrievement()
interpolation_strategy = FlatMapInterpolation()

augmentation_pipeline = NoAugmentation()

instance_level_objective = ConstantInstanceObjective()
instance_cost = ConstantInstanceCost()

surrogate_model = NoSurrogateModel()
training_strategy = NoTrainingStrategy()

selection_criteria = NoSelectionCriteria()
surrogate_sampler = RandomContinuousQuerySampler()
query_optimizer = NoQueryOptimizer()

knowledge_discovery_sampler = RandomContinuousQuerySampler()
knowledge_discovery_task = PrimScenarioDiscoveryKnowledgeDiscoveryTask()

evaluation_metrics = [RoundCounterEvaluator()]
