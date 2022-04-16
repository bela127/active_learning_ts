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

    def instantiate(self):
        return _instantiate_help(self)

    def _instantiate(self) -> T:
        if hasattr(self, '__orig_class__') and hasattr(self, 'args'):
            return self.__orig_class__.__args__[0](**self.args)
        else:
            return self.klass(**self.__dict__)


def _instantiate_help(o):
    if isinstance(o, list):
        out = []
        for e in o:
            out.append(_instantiate_help(e))
        return out
    elif isinstance(o, dict):
        out = {}
        for k,v in o.items():
            out[k] = _instantiate_help(v)
        return out
    elif isinstance(o, BlueprintElement):
        out = BlueprintElement()
        if hasattr(o, 'klass'):
            out = BlueprintElement[o.klass]()
        for k, v in o.__dict__.items():
            setattr(out, k, _instantiate_help(v))
        return out._instantiate()
    else:
        return o
