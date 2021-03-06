# Active Learning Test Suit

Active learning test suite is a test suite for active learning algorithms.
Its main purpose is the evaluation of active learning algorithms.
For a given surrogate model, and a given knowledge discovery task, this test suite is designed to evaluate
both the surrogate model and the test.

## Used Sources

https://github.com/Project-Platypus/PRIM
https://github.com/quaquel/EMAworkbench
https://github.com/scikit-activeml/scikit-activeml


https://pubsonline.informs.org/doi/pdf/10.1287/opre.41.3.435

https://www.sciencedirect.com/science/article/pii/S0925231214008145

## Framework Overview
The goal of this framework is to evaluate active learning models, from the models themselves, to helper methods
such as querying strategies and trainings strategies.
What exactly is evaluated is defined by a list of Evaluators.
The main building blocks of the framework are the Data Source, Surrogate Model, Trainer, Knowledge Discovery Task, 
and Query Optimiser.
The Data source is responsible for providing data of the form (x,y), where x is for example the input of a function, and
y is the output.
The Surrogate Model is responsible for emulating the Data Source, in order to minimise the number of points queried from
it.
The Trainer, is then responsible for training the Surrogate Model.
The Query Optimiser is responsible for selecting beneficial queries to optimally sample the search space.
The Knowledge Discovery samples many queries from the Surrogate Model, in order to obtain some knowledge about the data.

## Defining a Test
In order to define a test, a blueprint file must be created.
This must contain the following variables ([Blueprint](./active_learning_ts/experiments/blueprint_instance.py)):

### Repeat

The number of times that this experiment should be run.

### Learning Steps

A learning step consists of a batch of queries that are queried from the data source.
A learning step is subdivided into four section: 'Query Selection', 'Data Retrievement', 'Model Training', 
'Knowledge Discovery' and 'Evaluation', in that order.

This specifies how many learning steps should be taken

### Num Knowledge Discovery Queries

The knowledge discovery task is executed after every learning step.
This parameter defines how many learning steps (data points) the Knowledge Discovery task should have.

### Data source

The data source stores/generates data, in order to answer batches queries.

### Selection Criteria

The Selection Criteria is responsible for evaluating how beneficial a query would be, while considering:
Cost, The surrogate Model, The knowledge Discovery Task.
A Selection Criteria can combine multiple Selection Criteria, for this reason, a Selection Criteria returns a 1-d tensor
(an n-dimensional vector) for each query that it is responsible for evaluating.

### Retrievement Strategy

The retrievement strategy takes a query, generated by  the Query optimizer, and generates a batch of valid queries
that will be queried from the data source. 
The size of a batch depends on the retrievement strategy

This would for example allow a discrete Data source to be accessed as if it were continuous.
The retrievement strategy is therefore also responsible for informing the Query Optimizer, about which queries are
possible for the query optimizer to make.

As an example: one could define a data source that is discrete, and chose a Retrievement Strategy that 
allows this data source to be accessed as if it were continuous.
The retrievement Strategy, would inform the query optimizer, that the data source is continuous.
This is done with the use of 'Pools',

### Query Optimizer

The job of the query optimizer is to consider previously queried points, and the pool of all possible queries to 
generate large batches of queries.
It will then evaluate these queries using the selection criteria, and only the best of them is queried from the data 
source.

### Interpolation Strategy

The retrievement strategy is asked one query, and returns a batch of queries.
These batches are then answered.
At this point, we have data of the form (Original query, [ many queries ], [many results]).
The Interpolation Strategy is therefore responsible for returning data of the for ([many queries], [many results]).
There are many ways with which this could be done, solutions range from ignoring the original query, to interpolating
the results (in which case the output lists would be of length one)

### Surrogate Model

The surrogate Model's main purpose is to simulate the given Data source.
Depending on the knowledge discovery task, the surrogate Model might not aim to perfectly simulate the data Source,
so it is imperative that this is chosen to fit the task.

### Training Strategy

The Training Strategy is responsible for analysing the Surrogate Model, the queries that have been asked and the knowledge
discovery task, in order to give feedback to the Surrogate Model on how it should train.

### Surrogate Sampler

The surrogate sampler is responsible for sampling points to be queried from the data source.
In order to select valid points, it utilises the pool provided by the Retrievement Strategy.

### Knowledge Discovery Sampler

Much like the Surrogate Sampler, the Knowledge Discovery sampler is responsible for sampling points to be queries from 
the Surrogate Model.
These results of these queries are then used for the discovery.

### Knowledge Discovery Task

The knowledge Discovery task e.g. finding the maxima of a function, or calculating the areas of high interest.

### Evaluation Metrics

There are many metrics that one might want to evaluate.
The evaluation metrics are given access to all data in the current experiment (even other evaluation metrics).
Examples of simple evaluation metrics, are round timers, and counters for the number of points queried.
Some evaluation metrics can take lots of time, signals are therefore used to notify evaluation metrics about what work 
is currently being done.

### Instance level Objective

not really implemented. Insert ConstantInstanceObjective()  as placeholder

### Instance Cost

not really implemented. Insert ConstantInstanceCost() as placeholder

### Augmentation Pipeline

not really implemented. Insert NoAugmentation() as placeholder

# Installation

## Poetry
For linux users, give the following command to install poetry

```sh
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

With poetry installed, you can clone this repository using 

```sh
git clone [link to repository]
```

and then while in the repository execute 

```sh
poetry install 
```