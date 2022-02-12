from active_learning_ts.active_learner import ActiveLearner
from active_learning_ts.data_blackboard import Blackboard
from active_learning_ts.data_instance import DataInstance
from active_learning_ts.data_retrievement.data_retriever import DataRetriever
from active_learning_ts.evaluation.evaluator import Evaluator
from active_learning_ts.experiments.blueprint_instance import BlueprintInstance
from active_learning_ts.knowledge_discovery.knowledge_discovery import KnowledgeDiscovery
from active_learning_ts.oracle import Oracle
from active_learning_ts.query_selection.query_selector import QuerySelector
from active_learning_ts.training.trainer import Trainer


class Experiment:
    """
    Builds an experiment based on the instructions obtained from the blueprint.

    An experiment consists of:
    A blackboard where data is stored
    An Oracle in order to query data from the data source
    Another Oracle in order to query data from the surrogate
    A trainer in order to train the surrogate
    An Evaluator in order to evaluate the Experiment, according to the given metrics
    """

    def __init__(self, experiment_blueprint: BlueprintInstance, experiment_number: int = 0) -> None:
        self.experiment_number = experiment_number
        self.blackboard: Blackboard = Blackboard()
        self.surrogate_blackboard: Blackboard = Blackboard()
        self.experiment_blueprint = experiment_blueprint
        self.setup(self.experiment_blueprint)

    def setup(self, experiment_blueprint: BlueprintInstance):
        self.repeat: int = experiment_blueprint.repeat
        self.learning_steps: int = experiment_blueprint.learning_steps

        experiment_blueprint.data_source.post_init(retrievement_strategy=experiment_blueprint.retrievement_strategy)
        experiment_blueprint.retrievement_strategy.post_init(pool=experiment_blueprint.data_source.possible_queries())

        self.data_retriever = DataRetriever(
            data_source=experiment_blueprint.data_source,
            augmentation_pipeline=experiment_blueprint.augmentation_pipeline,
            interpolation_strategy=experiment_blueprint.interpolation_strategy
        )
        experiment_blueprint.surrogate_model.post_init(self.data_retriever)

        experiment_blueprint.training_strategy.post_init(experiment_blueprint.surrogate_model)
        experiment_blueprint.selection_criteria.post_init(experiment_blueprint.surrogate_model,
                                                          experiment_blueprint.knowledge_discovery_task)
        experiment_blueprint.query_optimizer.post_init(experiment_blueprint.surrogate_model,
                                                       experiment_blueprint.selection_criteria,
                                                       experiment_blueprint.surrogate_sampler)
        experiment_blueprint.surrogate_sampler.post_init(experiment_blueprint.retrievement_strategy.get_query_pool())
        experiment_blueprint.knowledge_discovery_sampler.post_init(
            experiment_blueprint.retrievement_strategy.get_query_pool())

        self.oracle = Oracle(
            DataInstance,
            self.blackboard,
            self.data_retriever,
            experiment_blueprint.instance_cost,
            experiment_blueprint.instance_level_objective,
        )

        self.sg_oracle = Oracle(
            DataInstance,
            self.surrogate_blackboard,
            experiment_blueprint.surrogate_model,
            experiment_blueprint.instance_cost,
            experiment_blueprint.instance_level_objective,
        )
        self.query_selector = QuerySelector(
            self.blackboard,
            experiment_blueprint.query_optimizer,
            experiment_blueprint.selection_criteria,
            self.sg_oracle
        )

        # TODO also need surrogate blackboard?
        self.trainer = Trainer(
            self.blackboard,
            experiment_blueprint.training_strategy
        )

        self.evaluator = Evaluator(
            evaluation_metrics=experiment_blueprint.evaluation_metrics,
            blackboard=self.blackboard,
            blueprint=experiment_blueprint
        )

        self.knowledge_discovery = KnowledgeDiscovery(
            knowledge_discovery_task=experiment_blueprint.knowledge_discovery_task,
            surrogate_model=self.sg_oracle,
            num_queries=experiment_blueprint.num_knowledge_discovery_queries,
            surrogate_sampler=experiment_blueprint.knowledge_discovery_sampler
        )

        self.active_learner = ActiveLearner(self.oracle, self.query_selector, self.blackboard, self.trainer,
                                            self.knowledge_discovery, self.evaluator)

    def run(self):
        for i in range(self.learning_steps):
            self.active_learner.learning_step()
            if self.evaluator.get_end_of_experiment():
                break
        self.evaluator.signal_end_of_experiment()
