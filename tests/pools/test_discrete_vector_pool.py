import tensorflow as tf
from distribution_data_generation.data_sources.data_set_data_source import DataSetDataSource

from active_learning_ts.pools.discrete_vector_pool import DiscreteVectorPool
from active_learning_ts.pools.retrievement_strategies.nearest_neighbours_retreivement_strategy import \
    NearestNeighboursRetrievementStrategy


def test_get_elements():
    x = tf.constant([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [3.0, 2.0, 1.0], [10.0, 2.0, 3.0]])
    data_source = DataSetDataSource(data_values=x, data_points=x)
    
    retrievement_strategy = NearestNeighboursRetrievementStrategy(2)
    data_source.post_init(retrievement_strategy)

    retrievement_strategy.post_init(data_source.possible_queries())

    pool = DiscreteVectorPool(in_dim=3, queries=x, retrievement_strategy=retrievement_strategy)

    query = tf.constant([[1.0, 2.0, 2.0]])
    normalized = pool.normalize(query)

    tf.assert_equal(pool.get_elements_normalized(normalized), pool.get_elements(query))
