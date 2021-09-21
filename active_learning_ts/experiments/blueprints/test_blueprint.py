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

repeat = 2
learning_steps = 10

data_source = TestDataSource()
retrievement_strategy = ExactRetrievement()
augmentation_pipeline = NoAugmentation()

instance_level_objective = ConstantInstanceObjective()
instance_cost = ConstantInstanceCost()

query_optimizer = RandomQueryOptimizer()
selection_criteria = GreedySelection()
