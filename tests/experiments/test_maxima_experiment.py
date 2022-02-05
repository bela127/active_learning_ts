import tests.experiments.blueprints.maxima_knowledge_discovery_task as blueprint
from active_learning_ts.experiments.experiment_runner import ExperimentRunner


def test_find_maxima():
    """
    Not yet a test, Just a cool demo (FYI the maximum of the function is 10, so the printed value should be around -10)
    :return:
    """
    er = ExperimentRunner([blueprint], log=True)
    er.run()

    print([x.get_evaluation() for x in blueprint.evaluation_metrics])
