import active_learning_ts.experiments.blueprints.test_blueprint as test_blueprint
from active_learning_ts.experiments.experiment_runner import ExperimentRunner


def test_base_functionality():
    """compilation test"""
    blueprints = [test_blueprint.TestBluePrint]
    er = ExperimentRunner(blueprints)
    er.run()
