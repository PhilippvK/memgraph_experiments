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


class ExportFilter(IntFlag):
    NONE = 0
    SELECTED = auto()          # 1
    ISO = auto()               # 2
    FILTERED_IO = auto()       # 4
    FILTERED_COMPLEX = auto()  # 8
    INVALID = auto()           # 16
    ERROR = auto()             # 32
    ALL = SELECTED | ISO | FILTERED_IO | FILTERED_COMPLEX | INVALID | ERROR
