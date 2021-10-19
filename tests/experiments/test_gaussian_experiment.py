import tests.experiments.blueprints.gaussian_blueprint as gaussian_blueprint
from active_learning_ts.experiments.experiment_runner import ExperimentRunner
import tensorflow as tf


def test_gaussian_experiment():
    er = ExperimentRunner([gaussian_blueprint])
    er.run()

    test = [tf.random.uniform(shape=(3,), minval=-5.0, maxval=5.0) for _ in
            range(0, 10)]

    print(gaussian_blueprint.surrogate_model.uncertainty(test))
