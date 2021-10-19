from active_learning_ts.query_selection.query_optimizers.random_query_optimizer import RandomQueryOptimizer
from active_learning_ts.query_selection.selection_criterias.explore_selection_criteria import ExploreSelectionCriteria
from active_learning_ts.surrogate_models.gaussion_surrogate_model import GaussianSurrogateModel
from distribution_data_generation.data_sources.multi_gausian_data_source import MultiGausianDataSource
import tensorflow as tf


def test_gaussian_experiment():
    gsm = GaussianSurrogateModel()
    mgs = MultiGausianDataSource(1, 1, -100.0, 100.0)
    sc = ExploreSelectionCriteria(gsm)
    qo = RandomQueryOptimizer(dim=1, max_x=100, min_x=-100, selection_criteria=sc, num_tries=10)

    for i in range(0, 100):
        queries = qo.optimize_query_candidates(1)
        results = mgs.query(queries)[1]
        gsm.learn(queries, results)

    test = [tf.random.uniform(shape=(1,), minval=-100.0, maxval=100.0) for _ in
            range(0, 10)]

    print(gsm.uncertainty(test))
