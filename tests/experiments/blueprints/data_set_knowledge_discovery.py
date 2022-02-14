import tensorflow as tf
from distribution_data_generation.data_sources.data_set_data_source import DataSetDataSource

from active_learning_ts.data_retrievement.augmentation.no_augmentation import NoAugmentation
from active_learning_ts.data_retrievement.interpolation_strategies.flat_map_interpolation import \
    FlatMapInterpolation
from active_learning_ts.evaluation.evaluation_metrics.rounder_counter_evaluator import RoundCounterEvaluator
from active_learning_ts.experiments.blueprint_element import BlueprintElement
from active_learning_ts.instance_properties.costs.constant_instance_cost import ConstantInstanceCost
from active_learning_ts.instance_properties.objectives.constant_instance_objective import ConstantInstanceObjective
from active_learning_ts.knowledge_discovery.discover_tasks.prim.prim_scenario_discovery_knowledge_discovery_task import \
    PrimScenarioDiscoveryKnowledgeDiscoveryTask
from active_learning_ts.data_retrievement.retrievement_strategies.exact_retrievement import ExactRetrievement
from active_learning_ts.query_selection.query_optimizers.no_query_optimizer import NoQueryOptimizer
from active_learning_ts.query_selection.query_samplers.no_query_sampler import NoQuerySampler
from active_learning_ts.query_selection.query_samplers.random_query_sampler import RandomContinuousQuerySampler
from active_learning_ts.query_selection.selection_criterias.no_selection_criteria import NoSelectionCriteria
from active_learning_ts.surrogate_model.surrogate_models.no_surrogate_model import NoSurrogateModel
from active_learning_ts.training.training_strategies.no_training_strategy import NoTrainingStrategy

x = tf.random.uniform(shape=(1000, 2)) * 100
y = tf.convert_to_tensor([tf.constant([0.9]) if 50 <= a[0] <= 60 and 10 <= a[1] <= 20 else tf.constant([0.0]) for a in x])

repeat = 1

learning_steps = 1
num_knowledge_discovery_queries = 100

data_source = BlueprintElement[DataSetDataSource]({'data_points': x, 'data_values': y})
retrievement_strategy = BlueprintElement[ExactRetrievement]()
interpolation_strategy = BlueprintElement[FlatMapInterpolation]()

augmentation_pipeline = BlueprintElement[NoAugmentation]()

instance_level_objective = BlueprintElement[ConstantInstanceObjective]()
instance_cost = BlueprintElement[ConstantInstanceCost]()

surrogate_model = BlueprintElement[NoSurrogateModel]()
training_strategy = BlueprintElement[NoTrainingStrategy]()

selection_criteria = BlueprintElement[NoSelectionCriteria]()
surrogate_sampler = BlueprintElement[NoQuerySampler]()
query_optimizer = BlueprintElement[NoQueryOptimizer]()

knowledge_discovery_sampler = BlueprintElement[RandomContinuousQuerySampler]()
knowledge_discovery_task = BlueprintElement[PrimScenarioDiscoveryKnowledgeDiscoveryTask]()

evaluation_metrics = [BlueprintElement[RoundCounterEvaluator]()]
