from enum import StrEnum, auto


class AnalyzeMethods(StrEnum):
    WEIGHTED_DENSITY = auto()
    CONSECUTIVE_TAIL = auto()
    STABILITY_STATE = auto()
    SCORED_WEIGHTED_DENSITY = auto()


class NormalizeMethod(StrEnum):
    NONE = auto()
    LINALG = auto()
    LOG_LINALG = auto()
    MAX = auto()
