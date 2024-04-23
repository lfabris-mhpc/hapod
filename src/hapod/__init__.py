from importlib import resources
try:
    from hapod import *
except ModuleNotFoundError:
    # import hapod as tomllib
    pass

__version__ = "0.0.1"
