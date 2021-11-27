import tensorflow as tf

from active_learning_ts.knowledge_discovery.prim.prim_scenario_discovery_knowledge_discovery_task import \
    PrimScenarioDiscoveryKnowledgeDiscoveryTask
from active_learning_ts.pools.continuous_vector_pool import ContinuousVectorPool
from active_learning_ts.surrogate_models.gaussion_surrogate_model import GaussianSurrogateModel


def test_prim():
    x = tf.unstack(tf.random.uniform(shape=(1000, 2)) * 100)
    sm = GaussianSurrogateModel()

    y = [tf.constant(0.9) if 50 <= a[0] <= 60 and 10 <= a[1] <= 20 else tf.constant(0.0) for a in x]

    sm.learn(x, y)

    pkd = PrimScenarioDiscoveryKnowledgeDiscoveryTask()
    pkd.post_init(surrogate_model=sm, surrogate_pool=ContinuousVectorPool(dim=2, ranges=[[(0, 100)], [(0, 100)]]))

    pkd.learn()
    pkd.learn()
    pkd.learn()

    test = tf.constant([[51., 15.0], [90., 90.]])

    tf.assert_equal(pkd.uncertainty(test), tf.constant([.0, .0]))

