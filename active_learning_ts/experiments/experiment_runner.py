import traceback

from active_learning_ts.experiments.experiment import Experiment


class ExperimentRunner:
    """
    Runs each experiment. In the Case multiple Experiments are to be run
    """

    def __init__(self, experiment_blueprints, file: str = "data_log", log: bool = False, write_mode: str = 'w') -> None:
        self.experiment_blueprints = experiment_blueprints
        self.to_log = log
        if self.to_log:
            self.file = open(file, write_mode)

    def run(self):
        if self.to_log:
            self.file.write('{\n')
        experiment_list = []
        for experiment_blueprint in self.experiment_blueprints:
            experiment_list.append(Experiment(experiment_blueprint))

        for experiment in experiment_list:
            try:
                experiment.run()
                if self.to_log:
                    self.log(experiment)
            except Exception as e:
                print(traceback.format_exc())
                print(e)
        if self.to_log:
            self.file.write('\n}')
            self.file.close()

    def log(self, experiment, index: int = 0):
        # TODO: add the other stuff
        ex = '{\n' + '"data_blackboard" : ' \
             + experiment.active_learner.blackboard.get_blackboard() \
             + ',\n"evaluations" : ' + experiment.active_learner.evaluator.get_evaluations().__str__() \
             + '}\n'
        p = '"' + experiment.experiment_blueprint.__name__ + str(index) + '" : ' + ex

        self.file.write(p)
