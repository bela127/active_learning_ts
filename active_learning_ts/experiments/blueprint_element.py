from dataclasses import dataclass, field
from typing import Generic, TypeVar, Type, Optional

T = TypeVar('T')


@dataclass
class BlueprintElement(Generic[T]):
    klass: Optional[Type] = field(init=False, repr=False)

    def __init__(self, args=None):
        if args is None:
            args = {}
        self.args = args

    def instantiate(self) -> T:
        if hasattr(self, '__orig_class__') and hasattr(self, 'args'):
            return self.__orig_class__.__args__[0](**self.args)
        else:
            return self.klass(**self.__dict__)
