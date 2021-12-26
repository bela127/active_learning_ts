import tensorflow as tf
from distribution_data_generation.data_sources.data_set_data_source import DataSetDataSource

from active_learning_ts.pools.discrete_vector_pool import DiscreteVectorPool
from active_learning_ts.pools.retrievement_strategies.nearest_neighbours_retreivement_strategy import \
    NearestNeighboursFindStrategy


def test_get_elements():
    find_strategy = NearestNeighboursFindStrategy(2)
    x = tf.constant([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [3.0, 2.0, 1.0], [10.0, 2.0, 3.0]])
    source = DataSetDataSource(in_dim=3, data_values=x, data_points=x)
    retrievement_strategy = NearestNeighboursFindStrategy(1)
    source.post_init(retrievement_strategy)
    retrievement_strategy.post_init(source.possible_queries())

    find_strategy.post_init(source.possible_queries())
    pool = DiscreteVectorPool(in_dim=3, queries=x, find_streategy=find_strategy)

    query = tf.constant([[1.0, 2.0, 2.0]])
    normalized = pool.normalize(query)

    tf.assert_equal(pool.get_elements_normalized(normalized), pool.get_elements(query))
