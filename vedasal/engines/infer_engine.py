from vedacore.misc import registry

from .base_engine import BaseEngine


@registry.register_module('engine')
class InferEngine(BaseEngine):
    pass
