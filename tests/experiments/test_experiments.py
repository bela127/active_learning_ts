from active_learning_ts.experiments.experiment_runner import ExperimentRunner
import active_learning_ts.experiments.blueprints.test_blueprint as test_blueprint


def test_base_functionality():
    blueprints = [test_blueprint]
    er = ExperimentRunner(blueprints)
    er.run()
    pass

#evaluators
