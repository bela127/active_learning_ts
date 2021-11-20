import tensorflow as tf

from active_learning_ts.experiments.experiment_runner import ExperimentRunner
from tests.experiments.blueprints import data_set_blueprint


def test_basic_functionality():
    er = ExperimentRunner([data_set_blueprint])
    er.run()

    test = [tf.random.uniform(shape=(3,), minval=-5.0, maxval=5.0) for _ in
            range(0, 10)]

    print(test)
    print(data_set_blueprint.surrogate_model.uncertainty(test))


