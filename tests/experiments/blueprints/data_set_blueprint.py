from distribution_data_generation.data_sources.data_set_data_source import DataSetDataSource

from active_learning_ts.data_retrievement.augmentation.no_augmentation import NoAugmentation
from active_learning_ts.data_retrievement.interpolation.interpolation_strategies.flat_map_interpolation import \
    FlatMapInterpolation
from active_learning_ts.evaluation.evaluation_metrics.avg_round_time_evaluator import AvgRoundTimeEvaluator
from active_learning_ts.evaluation.evaluation_metrics.rounder_counter_evaluator import RoundCounterEvaluator
from active_learning_ts.instance_properties.costs.constant_instance_cost import ConstantInstanceCost
from active_learning_ts.instance_properties.objectives.constant_instance_objective import ConstantInstanceObjective
from active_learning_ts.pools.find_strategies.exact_retrievement import ExactRetrievement
from active_learning_ts.pools.find_strategies.nearest_neighbours_retreivement_strategy import NearestNeighboursFindStrategy
from active_learning_ts.query_selection.query_optimizers.random_query_optimizer import RandomQueryOptimizer
from active_learning_ts.query_selection.selection_criterias.explore_selection_criteria import ExploreSelectionCriteria
from active_learning_ts.surrogate_models.gaussion_surrogate_model import GaussianSurrogateModel
from active_learning_ts.training.training_strategies.direct_training_strategy import DirectTrainingStrategy
import tensorflow as tf

repeat = 2
learning_steps = 10

x = []
y = []
for i in range(1, 100):
    x.append(tf.random.uniform(shape=(3,), minval=-5, maxval=5, seed=i))
    y.append(tf.random.uniform(shape=(3,), minval=-10, maxval=50, seed=i + 100))

retrievement_strategy = NearestNeighboursFindStrategy(num_neighbours=3)
data_source = DataSetDataSource(in_dim=3, retreivement_strategy=retrievement_strategy, data_points=x, data_values=y)

interpolation_strategy = FlatMapInterpolation()

augmentation_pipeline = NoAugmentation()

instance_level_objective = ConstantInstanceObjective()
instance_cost = ConstantInstanceCost()

surrogate_model = GaussianSurrogateModel()
training_strategy = DirectTrainingStrategy()

selection_criteria = ExploreSelectionCriteria()
query_optimizer = RandomQueryOptimizer(num_tries=10, shape=(3,))

evaluation_metrics = [AvgRoundTimeEvaluator(), RoundCounterEvaluator()]
