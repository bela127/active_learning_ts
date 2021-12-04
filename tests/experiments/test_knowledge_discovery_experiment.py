from active_learning_ts.experiments.experiment_runner import ExperimentRunner
from tests.experiments.blueprints import data_set_knowledge_discovery
import tensorflow as tf


def test_data_set_knowledge_discovery():
    er = ExperimentRunner([data_set_knowledge_discovery])
    er.run()

    test = tf.constant([[51., 15.0], [90., 90.]])

    tf.assert_equal(data_set_knowledge_discovery.knowledge_discovery_task.uncertainty(test), tf.constant([.0, .0]))
