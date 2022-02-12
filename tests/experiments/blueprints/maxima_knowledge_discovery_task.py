import tensorflow as tf
from distribution_data_generation.data_sources.data_set_data_source import DataSetDataSource

from active_learning_ts.data_retrievement.augmentation.no_augmentation import NoAugmentation
from active_learning_ts.data_retrievement.interpolation.interpolation_strategies.flat_map_interpolation import \
    FlatMapInterpolation
from active_learning_ts.evaluation.evaluation_metrics.total_knowledge_discovery_time_evaluator import \
    TotalKnowledgeDiscoveryTimeEvaluator
from active_learning_ts.evaluation.evaluation_metrics.total_query_time_evaluator import TotalQueryTimeEvaluator
from active_learning_ts.evaluation.evaluation_metrics.total_training_time_evaluator import TotalTrainingTimeEvaluator
from active_learning_ts.experiments.blueprint_element import BlueprintElement
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

repeat = 1

learning_steps = 10
num_knowledge_discovery_queries = 1000

x = tf.random.uniform((1000, 3), -10.0, 10.0)
y = tf.reshape(tf.convert_to_tensor(tf.reduce_min(x, 1)), (1000, 1))
y = 10 - (y * y)

data_source = BlueprintElement[DataSetDataSource]({'data_points': x, 'data_values': y})
retrievement_strategy = BlueprintElement[NearestNeighboursFindStrategy]({'num_neighbours': 5})
interpolation_strategy = BlueprintElement[FlatMapInterpolation]()

augmentation_pipeline = BlueprintElement[NoAugmentation]()

instance_level_objective = BlueprintElement[ConstantInstanceObjective]()
instance_cost = BlueprintElement[ConstantInstanceCost]()

surrogate_model = BlueprintElement[GaussianSurrogateModel]()
training_strategy = BlueprintElement[DirectTrainingStrategy]()

selection_criteria = BlueprintElement[ExploreSelectionCriteria]()
surrogate_sampler = BlueprintElement[RandomContinuousQuerySampler]()
query_optimizer = BlueprintElement[FixedValueOptimizer]({'value':0.3})

knowledge_discovery_sampler = BlueprintElement[RandomContinuousQuerySampler]()
knowledge_discovery_task = BlueprintElement[MaximaKnowledgeDiscoveryTask]()

evaluation_metrics = [BlueprintElement[TotalQueryTimeEvaluator](), BlueprintElement[TotalTrainingTimeEvaluator](),
                      BlueprintElement[TotalKnowledgeDiscoveryTimeEvaluator]()]
