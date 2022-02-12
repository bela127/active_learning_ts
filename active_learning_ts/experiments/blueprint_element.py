from typing import Generic, TypeVar

T = TypeVar('T')


class BlueprintElement(Generic[T]):
    def __init__(self, args=None):
        if args is None:
            args = {}
        self.args = args

    def instantiate(self) -> T:
        return self.__orig_class__.__args__[0](**self.args)
