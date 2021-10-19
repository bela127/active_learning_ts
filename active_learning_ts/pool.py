from typing import Protocol


class Pool(Protocol):
    """this is the pool of possible query candidates"""
    # TODO: It is totally possible that this is a constantly changing list. In order to select the nearest neighbor, a
    #  tree would have to be built every time a learning step has to be made. This might be ok since each learning step
    #  can have many queries. In any other case however, this would be horrible. However one could implement Static
    #  and dynamic retrievement strategies. Static retrievement strategies would make the assumption that the pool does
    #  not change, and gain efficiency. Dynamic RS would be a lot slower.
    pass
