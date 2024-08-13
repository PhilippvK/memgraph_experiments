def parse_llvm_const_str(val_str):
    assert val_str[-1] == "_"
    val_str = val_str[:-1]
    ty = None
    if " " in val_str:
        ty, val_str = val_str.split(" ", 1)
        valid_types = ["i64", "i32", "i16", "i8"]
        assert ty in valid_types, f"Unsupported const type: {ty}"
        # TODO: use type to decide on operand bits?
    assert " " not in val_str, f"Unexpected const syntax: {val_str}"
    val = float(val_str)
    assert int(val) == val
    val = int(val)
    # print("val", val)
    sign = True  # For now handle all constants as signed
    return val, ty, sign


def llvm_type_to_cdsl_type(llvm_type: str, signed: bool, reg_size=None):
    print("llvm_type_to_cdsl_type", llvm_type, signed, reg_size)
    if llvm_type is None:
        if reg_size is not None:
            if signed:
                return f"signed<{reg_size}>"
            else:
                return f"unsigned<{reg_size}>"
        else:
            raise RuntimeError(f"Unknown reg_size for unknown LLT: {llvm_type}")
    elif llvm_type == "LLT_invalid":
        if reg_size is not None:
            if isinstance(reg_size, str):
                if reg_size == "unknown":
                    raise ValueError("Unknown regsize!")
                else:
                    reg_size = int(reg_size)
            if signed:
                return f"signed<{reg_size}>"
            else:
                return f"unsigned<{reg_size}>"
        else:
            raise RuntimeError(f"Unknown reg_size for invalid LLT: {llvm_type}")
    else:
        llt_lookup = {
            "p0": (reg_size, 1, signed),
            "s32": (32, 1, signed),
            "s64": (32, 1, signed),
        }
        found = llt_lookup.get(llvm_type, None)
        assert found is not None, f"Lookup of LLT failed: {llvm_type}"
        sz, num_elements, signed_ = found
        if signed_:
            return f"signed<{sz}>"
        else:
            return f"unsigned<{sz}>"
    assert False, "unreachable"
