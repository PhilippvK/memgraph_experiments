import pickle
import logging
import argparse

# from pathlib import Path

import yaml
from tqdm import tqdm
import networkx as nx
import networkx.algorithms.isomorphism as iso


# from .iso import calc_io_isos
from .graph_utils import graph_to_file
from .hash import add_hash_attr

logger = logging.getLogger("combine_index")

NEUTRAL_ELEMENTS = {
    "ADD": [(None, 0), (0, None)],
    "ADDI": [(None, 0), (0, None)],
    "SUB": [(None, 0)],
    "MUL": [(None, 1), (1, None)],
    "OR": [(None, 0), (0, None)],
    "ORI": [(None, 0), (0, None)],
    "SLL": [(None, 0)],
    "SLLI": [(None, 0)],
    # "SRL"
    # "SRLI"
    # "SRA"
    # "SRAI"
    # "XOR"
    # "XOR"
    # ...
}


def handle_cmdline():
    # TODO: add help messages
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("index", nargs="+", help="TODO")  # print if None
    parser.add_argument("--log", default="info", choices=["critical", "error", "warning", "info", "debug"], help="TODO")
    parser.add_argument("--output", "-o", default=None, help="TODO")  # print if None
    parser.add_argument("--drop", action="store_true", help="TODO")
    parser.add_argument("--graph", default=None, help="TODO")  # print if None
    parser.add_argument("--progress", action="store_true", help="TODO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper()))
    return args


def load_candidates(in_path, progress: bool = False):
    candidates = []
    candidate_io_subs = []

    logger.info("Loading input %s", in_path)
    with open(in_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    global_data = yaml_data["global"]
    global_properties_ = global_data["properties"]
    candidates_data = yaml_data["candidates"]
    path_ids = set()
    for candidate_data in tqdm(candidates_data, disable=not progress):
        artifacts = candidate_data["artifacts"]
        # properties = candidate_data["properties"]
        candidates.append(candidate_data)
        io_sub_path = artifacts.get("io_sub", None)
        assert io_sub_path is not None, "PKL for io_sub not found in artifacts"
        with open(io_sub_path, "rb") as f:
            io_sub = pickle.load(f)
        candidate_io_subs.append(io_sub)
    global_properties_[0]["candidate_count"] = len(path_ids)
    total_count = len(candidates)
    logger.info(f"Loaded {total_count} candidates...")
    return yaml_data, candidates, candidate_io_subs


def detect_sub_specializations(candidates, candidate_io_subs, progress: bool = False):
    sub_specializations = {}
    # WARN: symmetry can not be used here!
    for i, c in tqdm(enumerate(candidates), disable=not progress):
        print("i", i)
        # print("c", c)
        sub_data = c["properties"]
        num_nodes = sub_data["#Nodes"]
        input_nodes = sub_data["InputNodes"]
        input_names = sub_data["InputNames"]
        output_nodes = sub_data["OutputNodes"]
        # operand_nodes = sub_data["OperandNodes"]
        # operand_names = sub_data["OperandNames"]
        # print("sub_data", sub_data)
        io_sub = candidate_io_subs[i]
        # print("io_sub", io_sub)
        artifacts = c["artifacts"]
        flat_artifact = artifacts["flat"]
        print("flat_artifact", flat_artifact)
        # TODO: speedup using hashes

        specs = []
        for j, c_ in tqdm(enumerate(candidates), disable=not progress):
            if i == j:
                continue
            # TODO: skip if num consts does not match?
            sub_data_ = c_["properties"]
            num_nodes_ = sub_data_["#Nodes"]
            input_nodes_ = sub_data_["InputNodes"]
            output_nodes_ = sub_data_["OutputNodes"]
            if (num_nodes - num_nodes_) != 1:
                continue
            instrs = sub_data["Instrs"]
            nodes = sub_data["Nodes"]
            instrs_ = sub_data_["Instrs"]
            # print("1", instrs_-instrs)
            # print("2", instrs-instrs_)
            from collections import Counter

            def is_sublist(x, y):
                cx, cy = Counter(x), Counter(y)
                cy.subtract(cx)
                dropped = [instruction for instruction, count in cy.items() if count > 0]
                return min(cy.values()) >= 0, dropped

            is_sublist_, dropped_instr = is_sublist(instrs_, instrs)
            if not is_sublist_:
                continue

            assert len(dropped_instr) == 1
            dropped_instr = dropped_instr[0]
            print("j", j)
            print("instrs", instrs)
            print("instrs_", instrs_)
            print("dropped_instr", dropped_instr)
            possible_dropped_nodes = [
                nodes[idx] for idx, instruction in enumerate(instrs) if instruction == dropped_instr
            ]
            print("num_nodes", num_nodes)
            print("num_nodes_", num_nodes_)
            print("possible_dropped_nodes", possible_dropped_nodes)
            io_sub_ = candidate_io_subs[j]
            artifacts_ = c_["artifacts"]
            flat_artifact_ = artifacts_["flat"]
            print("flat_artifact_", flat_artifact_)
            with open(flat_artifact, "r") as f:
                flat = f.read()
            with open(flat_artifact_, "r") as f:
                flat_ = f.read()
            print("flat", flat)
            print("flat_", flat_)
            for dropped_node in possible_dropped_nodes:
                print("dropped_node", dropped_node)

                def try_remove_node(io_sub, to_remove, dropped_instr, input_nodes):
                    assert nx.is_weakly_connected(io_sub)
                    new = io_sub.copy()
                    ins = io_sub.in_edges(to_remove, data=True)
                    outs = io_sub.out_edges(to_remove, data=True)
                    print("ins", ins)
                    print("outs", outs)
                    # assert len(outs) == 1  # TODO: not required?
                    sorted_ins = sorted(ins, key=lambda x: x[2]["properties"]["op_idx"])
                    print("sorted_ins", sorted_ins)
                    sorted_in_nodes = [x[0] for x in sorted_ins]
                    print("sorted_in_nodes", sorted_in_nodes)
                    is_variable = [x in input_nodes for x in sorted_in_nodes]
                    print("is_variable", is_variable)
                    if not any(is_variable):
                        return None, {}

                    assignments = {}
                    if dropped_instr in ["ADD", "ADDI", "XOR", "XORI", "OR"]:
                        neutral_element = 0
                        assert len(sorted_in_nodes) == 2
                        lhs_node, rhs_node = sorted_in_nodes
                        lhs_variable, rhs_variable = is_variable
                        if lhs_variable:
                            # check if input is used elsewhere
                            temp = io_sub.out_edges(lhs_node)
                            print("temp", temp)
                            if len(temp) > 1:
                                return None, {}
                            assignments[lhs_node] = neutral_element
                            new.remove_node(to_remove)
                            for _, dst, data in outs:
                                new.add_edge(rhs_node, dst, **data)
                        elif rhs_variable:
                            # check if input is used elsewhere
                            temp = io_sub.out_edges(rhs_node)
                            print("temp", temp)
                            if len(temp) > 1:
                                return None, {}
                            assignments[rhs_node] = neutral_element
                            new.remove_node(to_remove)
                            for _, dst, data in outs:
                                new.add_edge(lhs_node, dst, **data)
                        else:
                            assert False, "Should not be reached"
                    elif dropped_instr in ["MUL"]:
                        neutral_element = 1
                        assert len(sorted_in_nodes) == 2
                        lhs_node, rhs_node = sorted_in_nodes
                        lhs_variable, rhs_variable = is_variable
                        if lhs_variable:
                            assignments[lhs_node] = neutral_element
                            new.remove_node(to_remove)
                            for _, dst, data in outs:
                                new.add_edge(rhs_node, dst, **data)
                        elif rhs_variable:
                            assignments[rhs_node] = neutral_element
                            new.remove_node(to_remove)
                            for _, dst, data in outs:
                                new.add_edge(lhs_node, dst, **data)
                        else:
                            assert False, "Should not be reached"
                    elif dropped_instr in ["AND", "ANDI"]:
                        neutral_element = 0xffffffff  # TODO: always 32 bits? Will LLVM match this?
                        assert len(sorted_in_nodes) == 2
                        lhs_node, rhs_node = sorted_in_nodes
                        lhs_variable, rhs_variable = is_variable
                        if lhs_variable:
                            assignments[lhs_node] = neutral_element
                            new.remove_node(to_remove)
                            for _, dst, data in outs:
                                new.add_edge(rhs_node, dst, **data)
                        elif rhs_variable:
                            assignments[rhs_node] = neutral_element
                            new.remove_node(to_remove)
                            for _, dst, data in outs:
                                new.add_edge(lhs_node, dst, **data)
                        else:
                            assert False, "Should not be reached"
                    elif dropped_instr in ["SRLI", "SRAI", "SLLI", "SRL", "SLL"]:
                        neutral_element = 0
                        assert len(sorted_in_nodes) == 2
                        lhs_node, rhs_node = sorted_in_nodes
                        _, rhs_variable = is_variable
                        if not rhs_variable:
                            return None, {}
                        assignments[rhs_node] = neutral_element
                        new.remove_node(to_remove)
                        for _, dst, data in outs:
                            new.add_edge(lhs_node, dst, **data)
                    elif dropped_instr in ["SUB"]:
                        neutral_element = 0
                        assert len(sorted_in_nodes) == 2
                        lhs_node, rhs_node = sorted_in_nodes
                        _, rhs_variable = is_variable
                        if not rhs_variable:
                            return None, {}
                        assignments[rhs_node] = neutral_element
                        new.remove_node(to_remove)
                        for _, dst, data in outs:
                            new.add_edge(lhs_node, dst, **data)
                    elif dropped_instr in ["DIVU"]:
                        neutral_element = 1
                        assert len(sorted_in_nodes) == 2
                        lhs_node, rhs_node = sorted_in_nodes
                        _, rhs_variable = is_variable
                        if not rhs_variable:
                            return None, {}
                        assignments[rhs_node] = neutral_element
                        new.remove_node(to_remove)
                        for _, dst, data in outs:
                            new.add_edge(lhs_node, dst, **data)
                    elif dropped_instr in ["SLTIU", "MULH"]:
                        # Currently unsupported. Maybe possible?
                        return None, {}
                    else:
                        raise RuntimeError(f"Unhandled Instruction for dropping: {dropped_instr}")
                    # print("assignments", assignments)
                    # input("?")
                    if not nx.is_weakly_connected(new):
                        return None, {}
                    return new, assignments

                print("io_sub", io_sub)
                temp_io_sub, assignments = try_remove_node(io_sub, dropped_node, dropped_instr, input_nodes)
                print("temp_io_sub", temp_io_sub)
                print("assignments", assignments)
                if temp_io_sub is None:
                    print("cont (illegal)")
                    # input("!!!")
                    continue

                def drop_io_nodes(io_sub, input_nodes, output_nodes):
                    assert nx.is_weakly_connected(io_sub)
                    new = io_sub.copy()
                    for input_node in input_nodes:
                        new.remove_node(input_node)
                    for output_node in output_nodes:
                        new.remove_node(output_node)
                    assert nx.is_weakly_connected(io_sub)
                    return new

                temp_sub = drop_io_nodes(temp_io_sub, input_nodes, output_nodes)
                print("temp_sub", temp_sub)
                assert temp_sub is not None
                sub_ = drop_io_nodes(io_sub_, input_nodes_, output_nodes_)
                print("sub_", sub_)
                assert sub_ is not None

                add_hash_attr(temp_sub, attr_name="hash_attr2", ignore_const=False, ignore_names=False)
                add_hash_attr(sub_, attr_name="hash_attr2", ignore_const=False, ignore_names=False)
                # nm = iso.categorical_node_match("hash_attr_ignore_const", None)
                nm = iso.categorical_node_match("hash_attr2", None)
                em = iso.categorical_edge_match("hash_attr2", None)
                matcher = nx.algorithms.isomorphism.DiGraphMatcher(temp_sub, sub_, node_match=nm, edge_match=em)
                check = matcher.is_isomorphic()
                print("check", check)
                if not check:
                    print("cont (not iso)")
                    continue

                # TODO: (earlier) check if dropped node is a source or sink (ignoring I/Os)
                # TODO: check if temp_sub and sub_ are isomorph
                # TODO: normalize IO
                # TODO: check if temp_io_sub and io_sub_ are isomorph
                # TODO: define neural_elements (if required)

                # input(">>>")
                def get_input_name(node):
                    assert node in input_nodes
                    name = input_names[input_nodes.index(node)]
                    return name
                # def get_operand_name(node):
                #     assert node in operand_nodes
                #     name = operand_names[operand_nodes.index(node)]
                #     return name

                # assignments_str = " & ".join([f"{get_operand_name(x)} -> {y}" for x, y in assignments.items()])
                assignments_str = " & ".join([f"{get_input_name(x)} -> {y}" for x, y in assignments.items()])
                spec = (j, f"{dropped_instr} [{dropped_node}] {assignments_str}")
                # print("spec", spec)
                # input(">>>")
                specs.append(spec)

            # nm = iso.categorical_node_match("hash_attr_ignore_const", None)
            # em = iso.categorical_edge_match("hash_attr", None)
            # matcher = nx.algorithms.isomorphism.DiGraphMatcher(io_sub, io_sub_, node_match=nm, edge_match=em)
            # check = matcher.is_isomorphic()
            # if check:
            #     # print("check", check)
            #     print("======")
            #     # print("j", j)
            #     print("i,j", i, j)
            #     # print("c_", c_)
            #     # print("sub_data_", sub_data_)
            #     # print("flat_artifact_", flat_artifact_)
            #     mapping = matcher.mapping
            #     print("mapping", mapping)
        if len(specs) > 0:
            sub_specializations[i] = specs
    print("sub_specializations", sub_specializations)
    return sub_specializations


def get_spec_graph(sub_specializations, candidates):
    spec_graph = nx.DiGraph()
    for i, c in enumerate(candidates):
        spec_graph.add_node(i, label=f"c{i}")
    for i, specs in sub_specializations.items():
        for spec in specs:
            j, label = spec
            spec_graph.add_edge(i, j, label=label)
    return spec_graph


def get_sources(spec_graph):
    sources = [x for x in spec_graph.nodes() if spec_graph.in_degree(x) == 0]
    for source in sources:
        spec_graph.nodes[source].update({"style": "filled", "fillcolor": "lightgray"})
    return sources


def main():
    args = handle_cmdline()
    INS = args.index
    assert len(INS) == 1
    in_path = INS[0]
    OUT = args.output
    DROP = args.drop
    GRAPH = args.graph

    yaml_data, candidates, candidate_io_subs = load_candidates(in_path, progress=args.progress)

    sub_specializations = detect_sub_specializations(candidates, candidate_io_subs, progress=args.progress)
    for i, specs in sub_specializations.items():
        print(f"{i}\t: {len(specs)}")

    spec_graph = get_spec_graph(sub_specializations, candidates)
    print("spec_graph", spec_graph)
    if GRAPH is not None:
        graph_to_file(spec_graph, GRAPH)

    if DROP:
        sources = get_sources(spec_graph)
        print("sources", sources, len(sources))
        candidates = [x for i, x in enumerate(candidates) if i in sources]

    yaml_data["candidates"] = candidates
    if OUT:
        with open(OUT, "w") as f:
            yaml.dump(yaml_data, f)
    else:
        yaml_str = yaml.dump(yaml_data)
        print(yaml_str)


if __name__ == "__main__":
    main()
