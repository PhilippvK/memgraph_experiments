from enum import IntFlag, auto
from functools import reduce


def parse_enum_intflag(arg, cls):
    if arg is None:
        return None
        # assert hasattr(cls, "NONE")
        # return cls.NONE
    if isinstance(arg, cls):
        return arg
    if isinstance(arg, int):
        return cls(arg)
    not_allowed = [",", "(", ")", ";", "&"]
    for na in not_allowed:
        assert na not in arg, f"{na} not allowed in arg"
    splitted = arg.split("|")

    def helper(x):
        if x[0] == "~":
            return ~helper(x[1:])
        return cls[x]

    res = list(map(helper, splitted))
    reduced = reduce(lambda x, y: x | y, res)
    return reduced


class ExportFormat(IntFlag):
    NONE = 0
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
    HTML = auto()  # 1024
    MARKDOWN = auto()  # 2048
    # TREE = auto()  # 2048
    # TREE_PKL = auto()  # 4096


class ExportFilter(IntFlag):
    NONE = 0
    SELECTED = auto()  # 1
    ISO = auto()  # 2
    FILTERED_IO = auto()  # 4
    FILTERED_COMPLEX = auto()  # 8
    FILTERED_SIMPLE = auto()  # 16
    FILTERED_PRED = auto()  # 32
    FILTERED_ENC = auto()  # 64
    FILTERED_WEIGHTS = auto()  # 128
    FILTERED_MEM = auto()  # 256
    FILTERED_BRANCH = auto()  # 512
    INVALID = auto()  # 1024
    ERROR = auto()  # 2048
    # TODO: reorder
    FILTERED_OPERANDS = auto()  # 4096
    SYNTAX_ERROR = auto()  # 8192
    FILTERED_REGS = auto()  # 16384
    ALL = (
        SELECTED
        | ISO
        | FILTERED_IO
        | FILTERED_COMPLEX
        | FILTERED_SIMPLE
        | FILTERED_PRED
        | FILTERED_ENC
        | FILTERED_WEIGHTS
        | FILTERED_MEM
        | FILTERED_BRANCH
        | INVALID
        | ERROR
        | FILTERED_OPERANDS
    )


class InstrPredicate(IntFlag):
    NONE = 0
    MAY_LOAD = auto()  # 1
    MAY_STORE = auto()  # 2
    IS_PSEUDO = auto()  # 4
    IS_RETURN = auto()  # 8
    IS_CALL = auto()  # 16
    IS_TERMINATOR = auto()  # 32
    IS_BRANCH = auto()  # 64
    HAS_UNMODELED_SIDE_EFFECTS = auto()  # 128
    IS_COMMUTABLE = auto()  # 256
    ALL = (
        MAY_LOAD
        | MAY_STORE
        | IS_PSEUDO
        | IS_RETURN
        | IS_CALL
        | IS_TERMINATOR
        | IS_BRANCH
        | HAS_UNMODELED_SIDE_EFFECTS
        | IS_COMMUTABLE
    )


class CDFGStage(IntFlag):
    NONE = 0
    STAGE_0 = 1  # post irtranslator
    STAGE_1 = 2  # post legalizer
    STAGE_2 = 4  # post regbankselect
    STAGE_3 = 8  # post instructionselect
    STAGE_4 = 16  # post fallback/iseldag
    STAGE_5 = 32  # post finalizeisel/expandpseudos


class Variation(IntFlag):
    NONE = 0
    REUSE_IO = auto()  # 1
    REG2IMM = auto()  # 2
    CONST2IMM = auto()  # 4
    CONST2REG = auto()  # 8
    AUTO_IMM = auto()  # 16


class Status(IntFlag):
    NONE = 0
    GOOD = auto()  # 1
    ERROR = auto()  # 2
    QUERY_TIMEOUT = auto()  # 4
    STAGE_TIMEOUT = auto()  # 8


class ImmStrategy(IntFlag):
    MIN_REQUIRED_BITS = auto()
    NEXT_SUPPORTED_BITS = auto()
    DEFAULT_BITS = auto()
    MAX_SUPPORTED_BITS = auto()
    MAX_AVAILABLE_BITS = auto()
