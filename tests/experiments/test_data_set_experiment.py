import tensorflow as tf

from active_learning_ts.experiments.experiment_runner import ExperimentRunner
from tests.experiments.blueprints.data_set_blueprint import DataSetBlueprint


def test_basic_functionality():
    er = ExperimentRunner([DataSetBlueprint])
    er.run()

    test = [tf.random.uniform(shape=(3,), minval=-5.0, maxval=5.0, seed=_) for _ in
            range(0, 10)]

    for i in er.blueprint_instance_list[0].surrogate_model.uncertainty(test):
        assert i < 1.0
