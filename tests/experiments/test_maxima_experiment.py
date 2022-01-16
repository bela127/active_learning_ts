import tests.experiments.blueprints.maxima_knowledge_discovery_task as blueprint
from active_learning_ts.experiments.experiment_runner import ExperimentRunner


def test_find_maxima():
    er = ExperimentRunner([blueprint])
    er.run()
