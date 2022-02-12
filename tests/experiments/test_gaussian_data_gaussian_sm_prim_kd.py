import tests.experiments.blueprints.gaussian_data_gaussian_sm_prim_kd as blueprint
from active_learning_ts.experiments.experiment_runner import ExperimentRunner


def test_gaussian_data_gaussian_sm_prim_kd():
    """
    Smoke test, of the entire framework. There is currently no way to write such a test
    """
    er = ExperimentRunner([blueprint])
    er.run()
