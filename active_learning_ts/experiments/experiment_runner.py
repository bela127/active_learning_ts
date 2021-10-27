from active_learning_ts.experiments.experiment import Experiment


class ExperimentRunner:
    """
    Runs each experiment. In the Case multiple Experiments are to be run
    """
    def __init__(self, experiment_blueprints) -> None:
        self.experiment_blueprints = experiment_blueprints

    def run(self):
        for experiment_blueprint in self.experiment_blueprints:
            experiment = self.setup_experiment(experiment_blueprint)
            experiment.run()

    def setup_experiment(self, experiment_blueprint):
        experiment = Experiment(experiment_blueprint)
        return experiment
