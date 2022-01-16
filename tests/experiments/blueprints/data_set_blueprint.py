import tensorflow as tf
from distribution_data_generation.data_sources.data_set_data_source import DataSetDataSource

from active_learning_ts.data_retrievement.augmentation.no_augmentation import NoAugmentation
from active_learning_ts.data_retrievement.interpolation.interpolation_strategies.flat_map_interpolation import \
    FlatMapInterpolation
from active_learning_ts.evaluation.evaluation_metrics.avg_round_time_evaluator import AvgRoundTimeEvaluator
from active_learning_ts.evaluation.evaluation_metrics.rounder_counter_evaluator import RoundCounterEvaluator
from active_learning_ts.instance_properties.costs.constant_instance_cost import ConstantInstanceCost
from active_learning_ts.instance_properties.objectives.constant_instance_objective import ConstantInstanceObjective
from active_learning_ts.knowledge_discovery.no_knowledge_discovery_task import NoKnowledgeDiscoveryTask
from active_learning_ts.pools.retrievement_strategies.nearest_neighbours_retreivement_strategy import \
    NearestNeighboursFindStrategy
from active_learning_ts.query_selection.query_optimizers.maximum_query_optimizer import MaximumQueryOptimizer
from active_learning_ts.query_selection.query_samplers.random_query_sampler import RandomContinuousQuerySampler
from active_learning_ts.query_selection.selection_criterias.explore_selection_criteria import ExploreSelectionCriteria
from active_learning_ts.surrogate_models.gaussion_surrogate_model import GaussianSurrogateModel
from active_learning_ts.training.training_strategies.direct_training_strategy import DirectTrainingStrategy

repeat = 2
learning_steps = 10
num_knowledge_discovery_queries = 0

x = []
y = []
for i in range(1, 100):
    x.append(tf.random.uniform(shape=(3,), minval=-5, maxval=5, seed=i))
    y.append(tf.random.uniform(shape=(3,), minval=-10, maxval=50, seed=i + 100))

retrievement_strategy = NearestNeighboursFindStrategy(num_neighbours=3)
data_source = DataSetDataSource(data_points=tf.convert_to_tensor(x), data_values=tf.convert_to_tensor(y))

interpolation_strategy = FlatMapInterpolation()

augmentation_pipeline = NoAugmentation()

instance_level_objective = ConstantInstanceObjective()
instance_cost = ConstantInstanceCost()

surrogate_model = GaussianSurrogateModel()
training_strategy = DirectTrainingStrategy()

selection_criteria = ExploreSelectionCriteria()
surrogate_sampler = RandomContinuousQuerySampler()
query_optimizer = MaximumQueryOptimizer(num_tries=10)

knowledge_discovery_sampler = RandomContinuousQuerySampler()
knowledge_discovery_task = NoKnowledgeDiscoveryTask()

evaluation_metrics = [AvgRoundTimeEvaluator(), RoundCounterEvaluator()]
