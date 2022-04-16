from dataclasses import dataclass, field
from typing import Generic, TypeVar, Type, Optional

T = TypeVar('T')


@dataclass
class BlueprintElement(Generic[T]):
    """
    When creating a blueprint, the elements of the blueprint must be of the type BlueprintElement.
    A BlueprintElement acts as a configuration object that can be used to instantiate an arbitrary number of identically
    configured objects.

    In order to create a BlueprintElement you can either directly instantiate an object of the type BlueprintElement
    using the constructor. This takes the type of class to be constructed as a generic parameter, as well as the
    arguments of its constructor as a dictionary. This approach does not allow for typechecking, but is recommended for
    small elements.

    Another way is to inherit BlueprintElement directly. The class to be constructed must then define the variable
    'klass' with the class from which objects should be instantiated, as well as other init parameters. These parameters
    must have the same name and type as the init params of the given klass.

    e.g.

    class ExampleElement:
        def __init__(self, example_param1: float, example_param2: float):
            ...

    @dataclass
    class ExampleElementConfig(BlueprintElement[ExampleElement]):
    example_param1: float
    example_param2: float
    klass = ExampleElement
    """
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
