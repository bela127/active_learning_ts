import tensorflow as tf

from active_learning_ts.pools.discrete_vector_pool import DiscreteVectorPool
from active_learning_ts.pools.find_strategies.nearest_neighbours_find_strategy import NearestNeighboursFindStrategy


def test_get_elements():
    find_strategy = NearestNeighboursFindStrategy(2)
    x = [tf.constant([1.0, 2.0, 3.0]), tf.constant([2.0, 2.0, 3.0]), tf.constant([3.0, 2.0, 1.0]),
         tf.constant([10.0, 2.0, 3.0])]
    find_strategy.post_init(x)
    pool = DiscreteVectorPool(in_dim=3, queries=x, find_streategy=find_strategy)

    query = [tf.constant([1.0, 2.0, 2.0])]
    normalized = pool.normalize(query)

    tf.assert_equal(pool.get_elements_normalized(normalized), pool.get_elements(query))
