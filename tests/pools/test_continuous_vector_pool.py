import tensorflow as tf

from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool


def test_get_element():
    pool = ContinuousVectorPool(dim=2, ranges=[[(0, 2), (3, 5)], [(-10, 20), (50, 60)]])

    index = tf.constant([.5, .8])
    result = pool.get_element_normalized(index)

    query = pool.normalize(result)

    tf.assert_equal(index, query)
    tf.assert_equal(tf.constant([3.0, 52.0]), result)

    b = pool.is_valid(tf.constant([-0.1, 61.0]))
    print(b)
