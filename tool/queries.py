from typing import List, Optional

from .enums import CDFGStage


def generate_func_query(session: str, func: str, fix_cycles: bool = True, stage: CDFGStage = CDFGStage.STAGE_3):
    ret = f"""MATCH p0=(n00:INSTR)-[r01:DFG]->(n01:INSTR)
WHERE n00.func_name = '{func}'
AND n00.session = "{session}"
AND n00.stage = {stage}
"""
    if fix_cycles:
        # PHI nodes sometimes create cycles which are not allowed,
        # hence we drop all ingoing edges to PHIs as their src MI
        # is automatically marked as OUTPUT anyways.
        ret += """AND n01.name != "PHI" AND n01.name != "G_PHI"
"""
    ret += "RETURN p0;"
    return ret


def generate_candidates_query(
    session: str,
    func: str,
    bb: Optional[str],
    min_path_length: int,
    max_path_length: int,
    max_path_width: int,
    ignore_names: List[str],
    ignore_op_types: List[str],
    min_nodes: Optional[int] = None,
    max_nodes: Optional[int] = None,
    shared_input: bool = False,
    shared_output: bool = True,
    stage: CDFGStage = CDFGStage.STAGE_3,
    limit: Optional[int] = None,
):
    if shared_input:
        starts = ["a"] * max_path_width
    else:
        starts = [f"a{i}" for i in range(max_path_width)]
    if shared_output:
        ends = ["b"] * max_path_width
    else:
        ends = [f"b{i}" for i in range(max_path_width)]
    paths = [f"p{i}" for i in range(max_path_width)]
    match_rows = [
        f"MATCH {paths[i]}=({starts[i]}:INSTR)-[:DFG*{min_path_length}..{max_path_length}]->({ends[i]}:INSTR)"
        for i in range(max_path_width)
    ]
    match_str = "\n".join(match_rows)
    session_conds = [f"{x}.session = '{session}'" for x in set(starts) | set(ends)]
    func_conds = [f"{x}.func_name = '{func}'" for x in set(starts) | set(ends)]
    if bb:
        bb_conds = [f"{x}.basic_block = '{bb}'" for x in set(starts) | set(ends)]
    else:
        bb_conds = []
    stage_conds = [f"{x}.stage = {stage}" for x in set(starts) | set(ends)]
    conds = session_conds + func_conds + bb_conds + stage_conds
    conds_str = " AND ".join(conds)

    def gen_filter(path):
        name_filts = [f"node.name != '{name}'" for name in ignore_names]
        op_type_filts = [f"node.op_type != '{op_type}'" for op_type in ignore_op_types]
        filts = name_filts + op_type_filts
        filts_str = " AND ".join(filts)
        return f"all(node in nodes({path}) WHERE {filts_str})"

    filters = [gen_filter(path) for path in paths]
    filters_str = " AND ".join(filters)
    return_str = ", ".join(paths)
    if max_path_width > 1:
        # order_by_str = "size(collections.union(" + ", ".join([f"nodes({path})" for path in paths]) + "))"
        #  + ", ".join([f"nodes({path})" for path in paths]) + "))"
        order_by_str = "size("
        for k, path in enumerate(paths):
            if k < (len(paths) - 1):
                order_by_str += f"collections.union(nodes({path}), "
            else:
                order_by_str += f"nodes({path})"
        order_by_str += ")" * len(paths)
    else:
        order_by_str = "size(nodes(p0))"
    ret = f"""{match_str}
WHERE {conds_str}
AND {filters_str}
"""
    if min_nodes is not None:
        ret += f"AND size(collections.union(nodes(p0), nodes(p1))) >= {min_nodes}\n"
    if max_nodes is not None:
        ret += f"AND size(collections.union(nodes(p0), nodes(p1))) <= {max_nodes}\n"
    ret += f"""
RETURN {return_str}
ORDER BY {order_by_str} desc
"""
    if limit is not None:
        ret += f"""LIMIT {limit}
"""
    return ret + ";"
#     return """
# MATCH p0=(a0:INSTR)-[:DFG*1..5]->(b:INSTR)
# MATCH p1=(a1:INSTR)-[:DFG*1..5]->(b:INSTR)
# WHERE b.session = 'isaac-demo-20241105T115303' AND a0.session = 'isaac-demo-20241105T115303' AND a1.session = 'isaac-demo-20241105T115303' AND b.func_name = 'crcu16' AND a0.func_name = 'crcu16' AND a1.func_name = 'crcu16' AND b.basic_block = '%bb.0' AND a0.basic_block = '%bb.0' AND a1.basic_block = '%bb.0' AND b.stage = 32 AND a0.stage = 32 AND a1.stage = 32
# // AND all(node in nodes(p0) WHERE node.name != 'G_PHI' AND node.name != 'PHI' AND node.name != 'COPY' AND node.name != 'PseudoCALLIndirect' AND node.name != 'PseudoLGA' AND node.name != 'Select_GPR_Using_CC_GPR' AND node.op_type != 'input' AND node.op_type != 'constant')
# // AND all(node in nodes(p1) WHERE node.name != 'G_PHI' AND node.name != 'PHI' AND node.name != 'COPY' AND node.name != 'PseudoCALLIndirect' AND node.name != 'PseudoLGA' AND node.name != 'Select_GPR_Using_CC_GPR' AND node.op_type != 'input' AND node.op_type != 'constant')
# WITH collections.union(nodes(p0), nodes(p1)) as all_nodes, p0, p1
# WHERE true
# AND all(node in all_nodes WHERE node.name != 'G_PHI' AND node.name != 'PHI' AND node.name != 'COPY' AND node.name != 'PseudoCALLIndirect' AND node.name != 'PseudoLGA' AND node.name != 'Select_GPR_Using_CC_GPR' AND node.op_type != 'input' AND node.op_type != 'constant')
# AND size(all_nodes) >= 1
# AND size(all_nodes) <= 1000
# 
# RETURN p0, p1 // , size(all_nodes) as num_nodes
# // ORDER BY size(all_nodes) asc
# """
