from active_learning_ts.data_retrievement.interpolation.interpolation_strategies.flat_map_interpolation import \
    FlatMapInterpolation
from active_learning_ts.evaluation.evaluation_metrics.rounder_counter_evaluator import RoundCounterEvaluator
from active_learning_ts.pools.retrievement_strategies.exact_retrievement import ExactRetrievement
from active_learning_ts.query_selection.query_optimizers.no_query_optimizer import NoQueryOptimizer
from active_learning_ts.query_selection.query_samplers.random_query_sampler import RandomQuerySampler
from active_learning_ts.query_selection.selection_criterias.no_selection_criteria import (
    NoSelectionCriteria,
)
from active_learning_ts.query_selection.query_optimizers.maximum_query_optimizer import (
    MaximumQueryOptimizer,
)
from active_learning_ts.data_retrievement.augmentation.no_augmentation import (
    NoAugmentation,
)

from active_learning_ts.instance_properties.costs.constant_instance_cost import (
    ConstantInstanceCost,
)
from active_learning_ts.instance_properties.objectives.constant_instance_objective import (
    ConstantInstanceObjective,
)
from active_learning_ts.data_retrievement.data_sources.test_data_source import (
    TestDataSource,
)
from active_learning_ts.surrogate_models.gaussion_surrogate_model import GaussianSurrogateModel
from active_learning_ts.surrogate_models.no_surrogate_model import NoSurrogateModel
from active_learning_ts.training.training_strategies.direct_training_strategy import DirectTrainingStrategy
from active_learning_ts.training.training_strategies.no_training_strategy import NoTrainingStrategy

repeat = 2
learning_steps = 10

data_source = TestDataSource()
retrievement_strategy = ExactRetrievement()
augmentation_pipeline = NoAugmentation()
interpolation_strategy = FlatMapInterpolation()

instance_level_objective = ConstantInstanceObjective()
instance_cost = ConstantInstanceCost()

surrogate_model = NoSurrogateModel(retrievement_strategy=retrievement_strategy)
training_strategy = NoTrainingStrategy()

surrogate_sampler = RandomQuerySampler()
query_optimizer = NoQueryOptimizer()
selection_criteria = NoSelectionCriteria()

knowledge_discovery_sampler = RandomQuerySampler()

evaluation_metrics = [RoundCounterEvaluator()]
