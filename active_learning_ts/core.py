# TODO: i dont understand what this is supposed to do

def existing_pool(pool):
    def pool_creation_strategy():
        return pool
    return pool_creation_strategy

def pool_creation(pool_creation_strategy):
    pool = pool_creation_strategy()
    return pool


def pool_query_select(pool, selection_strategy):
    best_query = selection_strategy(pool)
    return best_query


def query_syntesis_select(syntesis_strategy):
    best_query = syntesis_strategy()
    return best_query


def objective_function(instance):
    return distance_to_objective


def loss_function(instance):
    return loss


def pool_query_syntesis(pool_creation_strategy, selection_strategy):
    def syntesis_strategy():
        pool = pool_creation(pool_creation_strategy)
        best_query = pool_query_select(pool, selection_strategy)
        return best_query

    return syntesis_strategy

def query_continues_data_source(query):
    return query_result

def query_pool_data_source(pool_index):
    return query_result


def labeled_data():
    return query, query_result

def pool_data_source(X,Y):
    return pool_data_source

def continues_data_source():
    return continues_data_source