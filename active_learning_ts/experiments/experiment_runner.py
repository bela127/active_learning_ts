import traceback
from typing import List

from active_learning_ts.experiments.blueprint import Blueprint
from active_learning_ts.experiments.blueprint_instance import BlueprintInstance
from active_learning_ts.experiments.experiment import Experiment


class ExperimentRunner:
    """
    Runs each experiment. In the Case multiple Experiments are to be run
    """

    def __init__(self, experiment_blueprints: List[Blueprint], file: str = "data_log", log: bool = False,
                 write_mode: str = 'w') -> None:
        self.experiment_blueprints = experiment_blueprints
        self.to_log = log
        if self.to_log:
            self.file = open(file, write_mode)
        self.experiment_list: List[Experiment] = []
        self.blueprint_instance_list: List[BlueprintInstance] = []

    def run(self):
        self.experiment_list = []
        self.blueprint_instance_list = []
        for experiment_blueprint in self.experiment_blueprints:
            for i in range(experiment_blueprint.repeat):
                a = BlueprintInstance(experiment_blueprint)
                self.blueprint_instance_list.append(a)
                self.experiment_list.append(Experiment(a, i))

        for experiment in self.experiment_list:
            try:
                experiment.run()
            except Exception as e:
                print(traceback.format_exc())
                print(e)
        if self.to_log:
            self.log()

    def log(self):
        # TODO: add more things if necessary
        self.file.write('{\n')
        out = []
        for experiment in self.experiment_list:
            ex = '{\n"data_blackboard" : ' \
                 + experiment.active_learner.blackboard.get_blackboard() \
                 + ',\n"surrogate_blackboard" : ' + experiment.surrogate_blackboard.get_blackboard() \
                 + ',\n"evaluations" : ' + experiment.active_learner.evaluator.get_evaluations().__str__() \
                 + '}'
            out.append(
                '"' + type(experiment.experiment_blueprint).__name__ + str(experiment.experiment_number) + '" : ' + ex)

        out = ',\n'.join(out)
        self.file.write(out)

        self.file.write('\n}')
        self.file.close()
