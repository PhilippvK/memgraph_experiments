from .mir_utils import gen_mir_func


def generate_mir(sub, sub_data, topo, GF, name="result", desc=None):
    inputs = sub_data["InputNodes"]
    num_inputs = int(sub_data["#InputNodes"])
    outputs = sub_data["OutputNodes"]
    num_outputs = int(sub_data["#OutputNodes"])
    j = 0  # reg's
    codes = []
    for node in sorted(sub.nodes, key=lambda x: topo.index(x)):
        node_ = sub[node]
        code_ = node_["properties"]["inst"]
        code_ = code_.split(", debug-location", 1)[0]
        if code_[-1] != "_":
            code_ += "_"
        codes.append(code_)
    code = "\n".join(codes)
    for inp in inputs:
        node = GF.nodes[inp]
        inst = node["properties"]["inst"]
        op_type = node["properties"]["op_type"]
        if "=" in inst:
            name = f"%inp{j}:gpr"
            j += 1
            lhs, _ = inst.split("=", 1)
            lhs = lhs.strip()
            assert "gpr" in lhs
            code = code.replace(lhs, name)
        else:
            if inst.startswith("$x"):  # phys reg
                pass
            else:
                assert op_type == "constant"
                assert inst[-1] == "_"
                const = inst[:-1]
                val = int(const)

                def get_ty_for_val(val):
                    def get_min_pow(x):
                        assert x >= 0
                        max_pow = 6
                        for i in range(max_pow + 1):
                            # print("i", i)
                            pow_val = 2**i
                            # print("pow_val", pow_val)
                            if x < 2**pow_val:
                                return pow_val
                        assert False

                    if val < 0:
                        val *= -1
                    min_pow = get_min_pow(val)
                    return f"i{min_pow}"

                ty = get_ty_for_val(val)
                code = code.replace(" " + inst, f" {ty} " + const)  # TODO: buggy?
    for j, outp in enumerate(outputs):
        node = GF.nodes[outp]
        inst = node["properties"]["inst"]
        if "=" in inst:
            name = f"%outp{j}:gpr"
            lhs, _ = inst.split("=", 1)
            lhs = lhs.strip()
            assert "gpr" in lhs
            code = code.replace(lhs, name)
        else:
            pass  # TODO: assert?
    # TODO: handle bbs:
    is_branch = False
    if "bb." in code:
        is_branch = True
    code = "\n".join([line[:-1] if line.endswith("_") else line for line in code.splitlines()])
    if desc:
        if is_branch:
            desc += ", IsBranch"
    mir_code = gen_mir_func(name, code, desc=desc)
    return mir_code
