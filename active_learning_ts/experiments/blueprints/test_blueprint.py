from active_learning_ts.evaluation.evaluation_metrics.rounder_counter_evaluator import RoundCounterEvaluator
from active_learning_ts.query_selection.selection_criterias.greedy_selection import (
    GreedySelection,
)
from active_learning_ts.query_selection.query_optimizers.random_query_optimizer import (
    RandomQueryOptimizer,
)
from active_learning_ts.data_retrievement.augmentation.no_augmentation import (
    NoAugmentation,
)
from active_learning_ts.data_retrievement.retrievement_strategies.exact_retrievement import (
    ExactRetrievement,
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
from active_learning_ts.training.training_strategies.direct_training_strategy import DirectTrainingStrategy

repeat = 2
learning_steps = 0

data_source = TestDataSource()
retrievement_strategy = ExactRetrievement()
augmentation_pipeline = NoAugmentation()

instance_level_objective = ConstantInstanceObjective()
instance_cost = ConstantInstanceCost()

# TODO: implement test surrogate model that just returns random uncertainty values
surrogate_model = GaussianSurrogateModel()
training_strategy = DirectTrainingStrategy()

query_optimizer = RandomQueryOptimizer(num_tries=10, shape=(3,))
selection_criteria = GreedySelection()

evaluation_metrics = [RoundCounterEvaluator()]
