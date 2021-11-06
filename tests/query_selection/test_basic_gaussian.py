from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool
from active_learning_ts.query_selection.query_optimizers.random_query_optimizer import RandomQueryOptimizer
from active_learning_ts.query_selection.selection_criterias.explore_selection_criteria import ExploreSelectionCriteria
from active_learning_ts.surrogate_models.gaussion_surrogate_model import GaussianSurrogateModel
from distribution_data_generation.data_sources.multi_gausian_data_source import MultiGausianDataSource
import tensorflow as tf


def test_gaussian_experiment():
    gsm = GaussianSurrogateModel()
    mgs = MultiGausianDataSource(1, 1, -100.0, 100.0)
    sc = ExploreSelectionCriteria()
    qo = RandomQueryOptimizer(shape=(1,), num_tries=10)

    sc.post_init(surrogate_model=gsm)
    qo.post_init(GaussianSurrogateModel, sc, ContinuousVectorPool(dim=1, ranges=[[(-100, 100)]]))

    for i in range(0, 10):
        queries = qo.optimize_query_candidates(1)
        results = mgs.query(queries)[1]
        gsm.learn(queries, results)

    test = [tf.random.uniform(shape=(1,), minval=-100.0, maxval=100.0) for _ in
            range(0, 10)]

    print(gsm.uncertainty(test))
