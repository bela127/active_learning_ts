import tensorflow as tf
from distribution_data_generation.data_sources.data_set_data_source import DataSetDataSource
from distribution_data_generation.data_sources.power_data_source import PowerDataSource

from active_learning_ts.data_retrievement.augmentation.no_augmentation import NoAugmentation
from active_learning_ts.data_retrievement.interpolation.interpolation_strategies.flat_map_interpolation import \
    FlatMapInterpolation
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

repeat = 2
learning_steps = 10
num_knowledge_discovery_queries = 100

s = PowerDataSource(dim=4, power=-4)

x = tf.random.uniform((100, 4), -10, 10)
y = tf.reshape(tf.convert_to_tensor(tf.reduce_min(x, 1)), (100, 1))

data_source = DataSetDataSource(x, y)
retrievement_strategy = NearestNeighboursFindStrategy(5)
interpolation_strategy = FlatMapInterpolation()

augmentation_pipeline = NoAugmentation()

instance_level_objective = ConstantInstanceObjective()
instance_cost = ConstantInstanceCost()

surrogate_model = GaussianSurrogateModel()
training_strategy = DirectTrainingStrategy()

selection_criteria = ExploreSelectionCriteria()
surrogate_sampler = RandomContinuousQuerySampler()
query_optimizer = FixedValueOptimizer(0.3)

knowledge_discovery_sampler = RandomContinuousQuerySampler()
knowledge_discovery_task = MaximaKnowledgeDiscoveryTask()

evaluation_metrics = []
