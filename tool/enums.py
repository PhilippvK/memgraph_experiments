from enum import IntFlag, auto


class ExportFormat(IntFlag):
    TXT = auto()  # 1
    DOT = auto()  # 2
    PDF = auto()  # 4
    PNG = auto()  # 8
    CSV = auto()  # 16
    PKL = auto()  # 32
    MIR = auto()  # 64
    CDSL = auto()  # 128
    FLAT = auto()  # 256
    YAML = auto()  # 512


class ExportFilter(IntFlag):
    NONE = 0
    SELECTED = auto()  # 1
    ISO = auto()  # 2
    FILTERED_IO = auto()  # 4
    FILTERED_COMPLEX = auto()  # 8
    FILTERED_SIMPLE = auto()  # 16
    FILTERED_PRED = auto()  # 32
    INVALID = auto()  # 64
    ERROR = auto()  # 128
    ALL = SELECTED | ISO | FILTERED_IO | FILTERED_COMPLEX | FILTERED_SIMPLE | FILTERED_PRED | INVALID | ERROR


class InstrPredicate(IntFlag):
    NONE = 0
    MAY_LOAD = auto()  # 1
    MAY_STORE = auto()  # 2
    IS_BRANCH = auto()  # 4
    ALL = MAY_LOAD | MAY_STORE | IS_BRANCH
