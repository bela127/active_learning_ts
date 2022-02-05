import tensorflow as tf

import tests.experiments.blueprints.gaussian_blueprint as gaussian_blueprint
from active_learning_ts.experiments.experiment_runner import ExperimentRunner


def test_gaussian_experiment():
    """
    technically a random test. but that high of an uncertainty is impossible, unless something is wrong
    """
    er = ExperimentRunner([gaussian_blueprint.GaussianBlueprint])
    er.run()

    test = [tf.random.uniform(shape=(3,), minval=-5.0, maxval=5.0, seed=2) for _ in
            range(0, 10)]

    for i in er.blueprint_instance_list[0].surrogate_model.uncertainty(test):
        assert i < 1.0
