from typing import Optional
from enum import IntEnum, IntFlag, auto


class TypeKind(IntEnum):
    INTEGER = auto()
    FLOAT = auto()
    SCALAR = auto()


class RegisterClass(IntFlag):
    NONE = 0
    UNKNOWN = auto()  # 1
    GPR = auto()  # 2
    FPR = auto()  # 4
    CSR = auto()  # 8


def kind2char(kind: TypeKind):
    lookup = {
        TypeKind.INTEGER: "i",
        TypeKind.FLOAT: "f",
        TypeKind.SCALAR: "s",
    }
    res = lookup.get(kind, None)
    assert res is not None, "Lookup failed for kind: {kind}"
    return res


class LLVMType:

    def __init__(self, elem_size, num_elem: Optional[int] = None, kind: TypeKind = TypeKind.INTEGER):
        self.elem_size = elem_size
        self.num_elem = num_elem
        self.kind = kind

    @property
    def is_vector(self):
        return self.num_elem is not None  # TODO: >1?

    @property
    def is_scalar(self):
        return self.num_elem is None  # TODO: allow 1?

    @property
    def total_bits(self):
        ret = self.size
        if self.num_elem is not None:
            ret *= self.num_elem
        return ret

    def __repr__(self):
        ret = ""
        if self.is_vector:
            ret += "v{self.num_elem}"
        else:
            assert self.is_scalar
        char = kind2char(self.kind)
        ret += f"{char}{self.size}"
        return ret

    def to_cdsl(self, signed: bool = False):
        if self.kind in [TypeKind.INTEGER, TypeKind.SCALAR]:
            if self.is_scalar:
                if signed:
                    ret = "signed"
                else:
                    ret = "unsigned"
                assert self.size > 0
                ret += f"<{self.size}>"
            else:
                assert self.is_vector
                # TODO: check pow2?
                raise NotImplementedError("to_cdsl not implemented for vectors")
        elif self.kind == TypeKind.FLOAT:
            raise NotImplementedError("to_cdsl not implemented for float")
        else:
            raise NotImplementedError(f"to_cdsl not implemented for kind {self.kind}")
        return ret


class LLVMRegister:

    def __init__(self, size: Optional[int] = None, llvm_type: Optional[LLVMType] = None, reg_class: Optional[RegisterClass] = None):
        self.size = size
        self.llvm_type = llvm_type
        self.reg_class = reg_class

    def __repr__(self):
        return f"LLVMRegister({self.size}, {self.llvm_type}, {self.reg_class})"

    def to_cdsl(self):
        raise NotImplementedError
