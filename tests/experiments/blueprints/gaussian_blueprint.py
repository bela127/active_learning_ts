from distribution_data_generation.data_sources.multi_gausian_data_source import MultiGausianDataSource

from active_learning_ts.data_retrievement.augmentation.no_augmentation import NoAugmentation
from active_learning_ts.data_retrievement.retrievement_strategies.exact_retrievement import ExactRetrievement
from active_learning_ts.instance_properties.costs.constant_instance_cost import ConstantInstanceCost
from active_learning_ts.instance_properties.objectives.constant_instance_objective import ConstantInstanceObjective
from active_learning_ts.query_selection.query_optimizers.random_query_optimizer import RandomQueryOptimizer
from active_learning_ts.query_selection.selection_criterias.explore_selection_criteria import ExploreSelectionCriteria
from active_learning_ts.surrogate_models.gaussion_surrogate_model import GaussianSurrogateModel
from active_learning_ts.training.training_strategies.direct_training_strategy import DirectTrainingStrategy

repeat = 2
learning_steps = 100

# TODO, not having 0 between min and max causes a logic error. Investigate
data_source = MultiGausianDataSource(in_dim=3, out_dim=2, min_x=-5, max_x=5)
retrievement_strategy = ExactRetrievement(query_pool=None)

augmentation_pipeline = NoAugmentation()

instance_level_objective = ConstantInstanceObjective()
instance_cost = ConstantInstanceCost()

surrogate_model = GaussianSurrogateModel()
training_strategy = DirectTrainingStrategy()

selection_criteria = ExploreSelectionCriteria()
query_optimizer = RandomQueryOptimizer(max_x=5, min_x=-5, num_tries=10, shape=(3,))
