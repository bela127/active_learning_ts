import tests.experiments.blueprints.gaussian_data_gaussian_sm_prim_kd as blueprint
from active_learning_ts.experiments.experiment_runner import ExperimentRunner
import tensorflow as tf


def test_gaussian_data_gaussian_sm_prim_kd():
    """
    Smoke test, of the entire framework. There is currently no way to write such a test
    """
    er = ExperimentRunner([blueprint])
    er.run()

    test = tf.random.uniform(shape=(10, 2), minval=-5.0, maxval=5.0, seed=2)

    for i in blueprint.surrogate_model.uncertainty(test):
        assert i < 1.1

    for i in blueprint.knowledge_discovery_task.uncertainty(test):
        assert i < 1.1
