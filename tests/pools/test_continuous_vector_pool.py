import tensorflow as tf

from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool


def test_get_element():
    pool = ContinuousVectorPool(dim=2, ranges=[[(0, 2), (3, 4)], [(-10, 20), (50, 60)]])

    index = tf.constant([.1, .8])
    result = pool._get_element_normalized(index)
    print(result)

    query = pool.normalize(result)
    print(query)
