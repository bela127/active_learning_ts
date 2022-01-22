import traceback

from active_learning_ts.experiments.experiment import Experiment


class ExperimentRunner:
    """
    Runs each experiment. In the Case multiple Experiments are to be run
    """

    def __init__(self, experiment_blueprints) -> None:
        self.experiment_blueprints = experiment_blueprints

    def run(self):
        experiment_list = []
        for experiment_blueprint in self.experiment_blueprints:
            experiment_list.append(Experiment(experiment_blueprint))

        for experiment in experiment_list:
            try:
                experiment.run()
            except Exception as e:
                print(traceback.format_exc())
                print(e)
