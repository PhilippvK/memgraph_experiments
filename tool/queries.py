"""Bounded Memgraph candidate traversal queries."""
from typing import List, Optional

from .enums import CDFGStage
from .types import RegisterClass


def _literal(value):
    # Keep exported queries self-contained while escaping Cypher string literals.
    return "'" + str(value).replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t') + "'"


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def generate_func_query(
    session: str, func: str, fix_cycles: bool = True, stage: CDFGStage = CDFGStage.STAGE_3, cross_bb: bool = False
):
    rel_suffix = "" if cross_bb else ":DFG"
    ret = f"""MATCH p0=(n00:INSTR)-[r01{rel_suffix}]->(n01:INSTR)
WHERE n00.func_name = {_literal(func)}
AND n00.session = {_literal(session)}
AND n00.stage = {_integer('stage', stage)}
"""
    if fix_cycles:
        ret += "AND NOT n01.name IN ['PHI', 'G_PHI']\n"
    return ret + "RETURN p0;"


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
    use_bb_id: bool = True,
    simple_mode: bool = False,
    maxmiso_idx: Optional[int] = None,
    allowed_reg_classes: Optional[RegisterClass] = None,
):
    """Return width path slots (repetition permits narrower candidates).

    Lengths count edges. Paths are simple, scoped to the requested graph and
    connected through a shared endpoint. With both sharing flags false, widths
    above one are rejected to avoid disconnected Cartesian products. Simple mode
    selects one path. A finite limit streams a traversal-order sample before
    sorting that sample by node count; it is not a global largest-first search.
    None means exhaustive enumeration: use a server/transaction timeout as well.
    """
    low = _integer('min_path_length', min_path_length)
    high = _integer('max_path_length', max_path_length)
    width = _integer('max_path_width', max_path_width, 1)
    stage = _integer('stage', stage)
    if low > high:
        raise ValueError('min_path_length must not exceed max_path_length')
    for name, value in [('min_nodes', min_nodes), ('max_nodes', max_nodes), ('limit', limit), ('maxmiso_idx', maxmiso_idx)]:
        if value is not None:
            _integer(name, value)
    if min_nodes is not None and max_nodes is not None and min_nodes > max_nodes:
        raise ValueError('min_nodes must not exceed max_nodes')
    if simple_mode:
        width = 1
    if width > 1 and not (shared_input or shared_output):
        raise ValueError('Multiple paths require a shared input or output')
    if max_nodes is not None:
        high = min(high, max(0, max_nodes - 1))
    paths = [f'p{i}' for i in range(width)]
    returns = ', '.join(paths)
    if low > high or max_nodes == 0 or limit == 0:
        return 'RETURN ' + ', '.join(f'null AS {p}' for p in paths) + ' LIMIT 0;'

    bb_id = None
    if bb is not None and use_bb_id:
        try:
            bb_id = int(bb.split('.', 1)[1])
        except (ValueError, IndexError, AttributeError) as exc:
            raise ValueError('bb must have the form %bb.<integer> when use_bb_id=True') from exc

    def predicate(node):
        terms = [f'{node}:INSTR', f'{node}.session = {_literal(session)}', f'{node}.stage = {stage}']
        if func is not None:
            terms.append(f'{node}.func_name = {_literal(func)}')
        if bb is not None:
            terms.append(f'{node}.bb_id = {bb_id}' if use_bb_id else f'{node}.basic_block = {_literal(bb)}')
        if maxmiso_idx is not None:
            terms.append(f'{node}.maxmiso_idx = {maxmiso_idx}')
        for prop, values in [('name', ignore_names), ('op_type', ignore_op_types)]:
            if values:
                terms.append(f'NOT {node}.{prop} IN [{", ".join(_literal(v) for v in values)}]')
        if allowed_reg_classes is not None:
            known = {'gpr': RegisterClass.GPR, 'fpr': RegisterClass.FPR, 'fpr32': RegisterClass.FPR,
                     'fpr64': RegisterClass.FPR, 'csr': RegisterClass.CSR}
            names = ', '.join(_literal(n) for n, flag in known.items() if flag & allowed_reg_classes)
            reg = f'{node}.out_reg_class'
            allowed = f'{reg} IN [{names}]'
            if allowed_reg_classes & RegisterClass.UNKNOWN:
                allowed += f' OR {reg} IS NULL OR NOT {reg} IN [{", ".join(_literal(n) for n in known)}]'
            terms.append(f'({allowed})')
        return ' AND '.join(terms)

    # Anchor the shared sink and expand backwards in the common MISO case.
    reverse = shared_output and not shared_input
    anchor = 'b' if reverse else 'a'
    rows = [f'MATCH ({anchor}:INSTR)', 'WHERE ' + predicate(anchor)]
    union = None
    for i, path in enumerate(paths):
        start = 'a' if shared_input else f'a{i}'
        end = 'b' if shared_output else f'b{i}'
        if width == 1 or i == 0:
            if reverse:
                end = anchor
            else:
                start = anchor
        # DFS filters execute at every expansion, before path combinations form.
        # Memgraph includes the candidate node in partial; reject repeat visits.
        filt = predicate('node') + ' AND single(visited IN nodes(partial) WHERE visited = node)'
        if max_nodes is not None:
            partial_nodes = 'nodes(partial)' if union is None else f'collections.union({union}, nodes(partial))'
            filt += f' AND size({partial_nodes}) <= {max_nodes}'
        expansion = f'[:DFG*{low}..{high} (edge, node, partial | {filt})]'
        if reverse:
            rows.append(f'MATCH {path}=({end})<-{expansion}-({start}:INSTR)')
        else:
            rows.append(f'MATCH {path}=({start})-{expansion}->({end}:INSTR)')
        union = f'nodes({path})' if union is None else f'collections.union({union}, nodes({path}))'
        checks = []
        if max_nodes is not None:
            checks.append(f'size({union}) <= {max_nodes}')
        if i > 0 and shared_input != shared_output:
            endpoint = f'a{i}' if reverse else f'b{i}'
            previous = ('a' if reverse else 'b') + str(i - 1)
            checks.append(f'id({previous}) <= id({endpoint})')
        if checks:
            rows.append('WHERE ' + ' AND '.join(checks))
    rows.append(f'WITH {returns}, {union} AS all_nodes')
    if min_nodes is not None:
        rows.append(f'WHERE size(all_nodes) >= {min_nodes}')
    if limit is not None:
        rows.append(f'WITH {returns}, all_nodes LIMIT {limit}')
    rows.extend([f'RETURN {returns}', 'ORDER BY size(all_nodes) DESC;'])
    return '\n'.join(rows)
