from importlib.metadata import version

from .planit import Chain, MailType, Node, Parallel, Plan, SlurmArgs, Step

__version__ = version("planit")

__all__ = [
    "Chain",
    "MailType",
    "Node",
    "Parallel",
    "Plan",
    "SlurmArgs",
    "Step",
]
