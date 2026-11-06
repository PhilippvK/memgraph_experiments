"""Conditional RISC-V branches supported by generated SelectionDAG patterns."""
BRANCHES = {"BEQ": ("==", False), "BNE": ("!=", False),
            "BLT": ("<", True), "BGE": (">=", True),
            "BLTU": ("<", False), "BGEU": (">=", False)}
BRANCH_OFFSET = "offset"
