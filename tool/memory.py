"""Scalar integer memory operations supported by fused instruction generation."""
import re

# (access width, sign-extend). Signedness is irrelevant for a full-XLEN load.
LOADS = {"LB": (8, True), "LBU": (8, False), "LH": (16, True),
         "LHU": (16, False), "LW": (32, True), "LWU": (32, False), "LD": (64, False)}
STORES = {"SB": 8, "SH": 16, "SW": 32, "SD": 64}


def unsupported_memory(properties):
    """Reject special memory ordering, rather than silently erase its contract."""
    if any(properties.get(key, False) for key in ("isVolatile", "isAtomic", "isOrdered", "volatile", "atomic")):
        return True
    return bool(re.search(r'\b(?:volatile|atomic|unordered|acquire|release|seq_cst)\b',
                          properties.get("inst", "")))
