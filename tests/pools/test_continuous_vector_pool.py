import tensorflow as tf

from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool


def test_get_element():
    pool = ContinuousVectorPool(dim=1, ranges=[[(0, 2)]])

    index = tf.constant([.8])
    print(pool._get_element_normalized(index))

