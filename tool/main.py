import logging
from datetime import datetime
from typing import Dict
from collections import defaultdict

# import concurrent.futures

import numpy as np
import pandas as pd
from anytree import AnyNode

# import matplotlib.pyplot as plt

# from anytree.iterators import AbstractIter

from .enums import ExportFormat, ExportFilter, Variation
from .timing import MeasureTime
from .settings import Settings
from .cmdline import handle_cmdline
import tool.stages as stages

logger = logging.getLogger("main")


args = handle_cmdline()

# TODO: pass allowed imm width (5bit, 12bits)
# TODO: also allow non-sequential imm ranges? (i.e. 1,2,4,8,...)?

settings = Settings.initialize(args)

with MeasureTime("Settings Validation", verbose=settings.times):
    stages.validate_settings(settings)

settings.to_yaml_file(settings.out_dir / "settings.yml")


logger.info("Running queries...")
with MeasureTime("Connect to DB", verbose=settings.times):
    driver = stages.connect_db(settings)

with MeasureTime("Query func from DB", verbose=settings.times):
    func_results = stages.query_func(settings, driver)


with MeasureTime("Query candidates from DB", verbose=settings.times):
    results = stages.query_candidates(settings, driver)

# TODO: move to helper func
# TODO: print number of results
with MeasureTime("Conversion to NX (func)", verbose=settings.times):
    GF = stages.convert_func(func_results)


# TODO: move to helper and share code
with MeasureTime("Conversion to NX (candidates)", verbose=settings.times):
    G = stages.convert_candidates(results)


with MeasureTime("Subgraph Generation", verbose=settings.times):
    subs = stages.generate_subgraphs(settings, results, G)


# for i, result in enumerate(results):
#     print("result", result, i, dir(result), result.data())

with MeasureTime("Relabeling", verbose=settings.times):
    G, GF, topo = stages.relabel_nodes(settings, G, GF, subs)

index_artifacts = defaultdict(dict)


if settings.write.func:
    with MeasureTime("Dumping GF graph", verbose=settings.times):
        stages.dump_func_graph(settings, GF, index_artifacts)


# print("subs", subs, len(subs))
# isos = set()
# for i, sub in enumerate(subs):
#     print("sub", sub, sub.nodes)
#     nm = lambda x, y: x["label"] == y["label"]
#     isos_ = set(j for j, sub_ in enumerate(subs) if j > i and nx.is_isomorphic(sub, sub_, node_match=nm))
#     iso_count = len(isos_)
#     isos |= isos_
#     print("iso_count", iso_count)
# print("isos", isos, len(isos))


now = datetime.now()
ts = now.strftime("%Y%m%dT%H%M%S")


def init_global_df(settings, ts):
    global_df = pd.DataFrame()

    global_df["timestamp"] = [ts]
    global_df["min_inputs"] = [settings.filters.min_inputs]
    global_df["max_inputs"] = [settings.filters.max_inputs]
    global_df["min_outputs"] = [settings.filters.min_outputs]
    global_df["max_outputs"] = [settings.filters.max_outputs]
    global_df["min_nodes"] = [settings.filters.min_nodes]
    global_df["max_nodes"] = [settings.filters.max_nodes]
    global_df["xlen"] = [settings.riscv.xlen]
    global_df["session"] = [settings.query.session]
    global_df["func"] = [settings.query.func]
    global_df["bb"] = [settings.query.bb]
    global_df["stage"] = [settings.query.stage]
    global_df["limit_results"] = [settings.query.limit_results]
    global_df["min_path_len"] = [settings.query.min_path_len]
    global_df["max_path_len"] = [settings.query.max_path_len]
    global_df["max_path_width"] = [settings.query.max_path_width]
    global_df["instr_predicates"] = [settings.filters.instr_predicates]
    global_df["ignore_names"] = [settings.query.ignore_names]
    global_df["ignore_op_types"] = [settings.query.ignore_op_types]
    global_df["allowed_enc_sizes"] = [settings.filters.allowed_enc_sizes]
    global_df["max_enc_footprint"] = [settings.filters.max_enc_footprint]
    global_df["max_enc_weight"] = [settings.filters.max_enc_weight]
    global_df["min_enc_bits_left"] = [settings.filters.min_enc_bits_left]
    global_df["min_iso_weight"] = [settings.filters.min_iso_weight]
    global_df["max_loads"] = [settings.filters.max_loads]
    global_df["max_stores"] = [settings.filters.max_stores]
    global_df["max_mems"] = [settings.filters.max_mems]
    global_df["max_branches"] = [settings.filters.max_branches]
    # TODO: MIN_FREQ, MAX_INSTRS, MAX_UNIQUE_INSTRS
    # TODO: variations
    # TODO: add missing
    return global_df


def init_subs_df(settings):
    subs_df = pd.DataFrame({"result": list(range(len(subs)))})
    subs_df["DateTime"] = ts
    subs_df["Parent"] = np.nan  # used to find the original sub for a variation
    subs_df["Variations"] = [Variation.NONE] * len(subs_df)  # used to specify applied variations for Children
    subs_df["SubHash"] = None
    subs_df["IOSubHash"] = None
    subs_df["FullHash"] = None
    subs_df["GlobalHash"] = None
    subs_df["Isos"] = [np.array([])] * len(subs_df)
    subs_df["#Isos"] = np.nan
    subs_df["IsosNO"] = [np.array([])] * len(subs_df)
    subs_df["#IsosNO"] = np.nan
    subs_df["Nodes"] = [np.array([])] * len(subs_df)
    # subs_df["#Nodes"] = np.nan
    subs_df["InputNodes"] = [np.array([])] * len(subs_df)
    subs_df["#InputNodes"] = np.nan
    subs_df["InputNames"] = [np.array([])] * len(subs_df)
    subs_df["ConstantNodes"] = [np.array([])] * len(subs_df)
    subs_df["ConstantValues"] = [np.array([])] * len(subs_df)
    subs_df["ConstantSigns"] = [np.array([])] * len(subs_df)
    subs_df["ConstantMinBits"] = [np.array([])] * len(subs_df)
    subs_df["#ConstantNodes"] = np.nan  # TODO: analyze constants
    subs_df["OutputNodes"] = [np.array([])] * len(subs_df)
    subs_df["OutputNames"] = [np.array([])] * len(subs_df)
    subs_df["#OutputNodes"] = np.nan
    subs_df["#Operands"] = np.nan
    subs_df["OperandNames"] = [np.array([])] * len(subs_df)
    subs_df["OperandNodes"] = [np.array([])] * len(subs_df)
    subs_df["OperandDirs"] = [np.array([])] * len(subs_df)
    subs_df["OperandTypes"] = [np.array([])] * len(subs_df)
    subs_df["OperandRegClasses"] = [np.array([])] * len(subs_df)  # TODO
    subs_df["OperandEncBits"] = [np.array([])] * len(subs_df)
    subs_df["OperandEncBitsSum"] = np.nan
    subs_df["Constraints"] = [np.array([])] * len(subs_df)
    subs_df["Instrs"] = [np.array([])] * len(subs_df)
    subs_df["#Instrs"] = np.nan
    subs_df["#Loads"] = np.nan
    subs_df["#Stores"] = np.nan
    subs_df["#Mems"] = np.nan
    subs_df["#Terminators"] = np.nan
    subs_df["#Branches"] = np.nan
    subs_df["LoadNodes"] = [np.array([])] * len(subs_df)
    subs_df["StoreNodes"] = [np.array([])] * len(subs_df)
    subs_df["TerminatorNodes"] = [np.array([])] * len(subs_df)
    subs_df["BranchNodes"] = [np.array([])] * len(subs_df)
    subs_df["UniqueInstrs"] = [np.array([])] * len(subs_df)
    subs_df["#UniqueInstrs"] = np.nan
    for enc_size in settings.filters.allowed_enc_sizes:
        subs_df[f"EncodingBitsLeft ({enc_size} bits)"] = np.nan
        subs_df[f"EncodingWeight ({enc_size} bits)"] = np.nan
        subs_df[f"EncodingFootprint ({enc_size} bits)"] = np.nan
    subs_df["Weight"] = np.nan
    subs_df["Freq"] = np.nan
    subs_df["IsoNodes"] = [np.array([])] * len(subs_df)
    subs_df["IsoWeight"] = np.nan
    subs_df["Status"] = [ExportFilter.SELECTED] * len(subs_df)  # TODO: init with UNKNOWN
    # print("subs_df")
    # print(subs_df)
    return subs_df


global_df = init_global_df(settings, ts)
subs_df = init_subs_df(settings)


with MeasureTime("I/O Analysis", verbose=settings.times):
    io_subs = stages.analyze_io(settings, GF, subs, subs_df)


# with MeasureTime("Normalize Graphs", verbose=settings.times):
#     logger.info("Normalizing graphs...")
#     for i, io_sub in enumerate(tqdm(io_subs, disable=not settings.progress)):
#         print("i, io_sub", i, io_sub)
#         nodes = io_sub.nodes
#         sub_data = subs_df.iloc[i]
#         inputs = sub_data["InputNodes"]
#         # TODO: also handle constants
#         new_graph = nx.MultiDiGraph()
#         inputs_constants = inputs | constants
#         input_node_mapping = {n: f"src{i}" for i, n in enumerate(inputs)}
#         for n, data in io_sub.nodes(data=True):
#             new_label = input_node_mapping.get(n, n)  # Replace input nodes, keep others unchanged
#             new_graph.add_node(new_label, **data)
#         for dst in io_sub.nodes:
#             preds = list(io_sub.predecessors(dst))
#
#             if preds:
#                 if is_commutative(graph.nodes[dst]):
#                     preds.sort()  # Sort predecessors for commutative operations
#
#                 for src in preds:
#                     new_src = node_mapping.get(src, src)
#                     new_dst = node_mapping.get(dst, dst)
#                     new_graph.add_edge(new_src, new_dst)


with MeasureTime("SubHash Creation", verbose=settings.times):
    stages.create_hashes(settings, subs, io_subs, subs_df)


with MeasureTime("Isomorphism Check", verbose=settings.times):
    io_isos, sub_io_isos = stages.check_iso(settings, io_subs, subs_df)


with MeasureTime("Overlap Check", verbose=settings.times):
    stages.check_overlaps(subs, GF, subs_df, io_isos, sub_io_isos)


with MeasureTime("Predicate Detection", verbose=settings.times):
    stages.detect_predicates(settings, subs, subs_df, io_isos)


with MeasureTime("Register Detection", verbose=settings.times):
    stages.detect_registers(settings, subs, subs_df, io_isos)


# TODO: toggle on/off via cmdline?
with MeasureTime("Schedule Subs", verbose=settings.times):
    stages.schedule_subs(settings, io_subs, subs_df, io_isos)


# TODO: filter before iso check?
with MeasureTime("Filtering subgraphs", verbose=settings.times):
    stages.filter_subs(settings, subs, subs_df, io_isos)


with MeasureTime("Determining IO names", verbose=settings.times):
    stages.assign_io_names(settings, subs, io_subs, subs_df)


with MeasureTime("Analyze Constants", verbose=settings.times):
    stages.analyze_const(settings, subs, io_subs, subs_df)


with MeasureTime("Creating Constants Histograms", verbose=settings.times):
    stages.const_hist(settings, io_subs, subs_df)


with MeasureTime("Variation generation", verbose=settings.times):
    stages.generate_variations(settings, subs, GF, io_subs, subs_df)


with MeasureTime("FullHash Creation", verbose=settings.times):
    stages.create_fullhash(settings, subs, io_subs, subs_df, global_df)


with MeasureTime("Filtering subgraphs (Operands)", verbose=settings.times):
    stages.filter_subs_operands(settings, subs, subs_df)


# TODO: filter before iso check?
with MeasureTime("Filtering subgraphs (Encoding)", verbose=settings.times):
    stages.filter_subs_encoding(settings, subs, subs_df)


with MeasureTime("Filtering subgraphs (Weights)", verbose=settings.times):
    stages.filter_subs_weights(settings, subs, subs_df)


sub_stmts: Dict[int, AnyNode] = {}
if settings.write.tree_fmt:
    errs = set()
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.gen_flt.value) > 0].copy()
    with MeasureTime("Tree Generation", verbose=settings.times):
        stages.generate_tree(settings, subs, io_subs, subs_df, filtered_subs_df, index_artifacts, sub_stmts, errs)
    subs_df.loc[list(errs), "Status"] = ExportFilter.ERROR.value


if settings.write.gen:
    errs = set()
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.gen_flt.value) > 0].copy()
    if settings.write.gen_fmt & ExportFormat.MIR:
        with MeasureTime("MIR Generation", verbose=settings.times):
            stages.generate_mir(settings, subs, GF, subs_df, filtered_subs_df, index_artifacts, sub_stmts, errs, topo)
    if settings.write.gen_fmt & ExportFormat.CDSL:
        with MeasureTime("CDSL Generation", verbose=settings.times):
            stages.generate_cdsl(settings, subs, subs_df, filtered_subs_df, index_artifacts, sub_stmts, errs)
    if settings.write.gen_fmt & ExportFormat.FLAT:
        with MeasureTime("FLAT Generation", verbose=settings.times):
            stages.generate_flat(settings, subs, subs_df, filtered_subs_df, index_artifacts, sub_stmts, errs)
    subs_df.loc[list(errs), "Status"] = ExportFilter.ERROR.value


# def extract_inputs_and_constants(graph):
#     """Extracts input and constant nodes separately."""
#     inputs = set()
#     constants = {}
#     for node, data in graph.nodes(data=True):
#         if "value" in data:
#             constants[node] = data["value"]
#         elif data.get("type") == "input":
#             inputs.add(node)
#     return inputs, constants
#
#
# def extract_specialization_mapping(general, specialized, gen_inputs, spec_inputs, gen_constants, spec_constants):
#     """Extracts the mapping of input nodes replaced by constants during specialization."""
#     # TODO: merge with can_specialize?
#     mapping = {}
#     gen_inputs, gen_constants = extract_inputs_and_constants(general)
#     spec_inputs, spec_constants = extract_inputs_and_constants(specialized)
#
#     for node in gen_inputs:
#         if node not in spec_inputs and node in spec_constants:
#             mapping[node] = spec_constants[node]  # Input replaced by constant
#     return mapping
#
#
# def can_specialize(general_graph, specialized_graph, general_inputs, specialized_inputs,
#                    general_constants, specialized_constants):
#     """
#     Checks if `specialized_graph` is a specialization of `general_graph`,
#     meaning some inputs in `general_graph` have been replaced by constants.
#     """
#     # general_inputs, general_constants = extract_inputs_and_constants(general_graph)
#     # print("general_inputs", general_inputs)
#     # print("general_constants", general_constants)
#     # specialized_inputs, specialized_constants = extract_inputs_and_constants(specialized_graph)
#     # print("specialized_inputs", specialized_inputs)
#     # print("specialized_constants", specialized_constants)
#
#     # Specialization must have the same or fewer inputs and possibly more constants
#     if not specialized_inputs.issubset(general_inputs):
#         # print("ret if1")
#         return False
#
#     # Construct a mapping from general -> specialized
#     # TODO: why is this not used?
#     # input_mappings = {inp: inp for inp in specialized_inputs}  # Inputs must map 1:1
#     replaced_constants = {}
#
#     for node in general_inputs:
#         if node not in specialized_inputs:
#             if node in specialized_constants:  # Input replaced by constant?
#                 replaced_constants[node] = specialized_constants[node]
#             else:
#                 # print("ret invalid placement")
#                 return False  # Removed input without a valid replacement
#     # print("replaced_constants", replaced_constants)
#
#     # Check if computational structure remains the same
#     general_copy = general_graph.copy()
#     specialized_copy = specialized_graph.copy()
#
#     # Apply replacements in general graph
#     for node, value in replaced_constants.items():
#         general_copy.nodes[node]["value"] = value
#         general_copy.nodes[node]["type"] = "constant"
#
#     # Now, check for graph isomorphism (same structure after replacements)
#     matcher = nx.algorithms.isomorphism.GraphMatcher(
#         general_copy, specialized_copy,
#         node_match=lambda d1, d2: d1.get("op") == d2.get("op") and d1.get("value", None) == d2.get("value", None)
#     )
#     is_iso = matcher.is_isomorphic()
#     # print("is_iso", is_iso)
#
#     return is_iso
#
#
# with MeasureTime("Generate Specialization Graph", verbose=settings.times):
#     # ? = stages.generate_spec(?)
#     ### TODO ###
#     logger.info("Generating specialization graph...")
#     filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.sub.flt.value) > 0].copy()
#     io_subs_iter = [(i, io_sub) for i, io_sub in enumerate(io_subs) if i in filtered_subs_df.index]
#     spec_graph = nx.DiGraph()
#     for i, io_sub in io_subs_iter:
#         spec_graph.add_node(i, graph=io_sub)
#
#     for i, io_sub in tqdm(io_subs_iter, disable=not settings.progress):
#         # sub = subs[i]
#         # nodes = sub.nodes
#         sub_data = subs_df.iloc[i]
#         inputs = set(sub_data["InputNodes"])
#         constant_nodes = sub_data["ConstantNodes"]
#         # print("constant_nodes", constant_nodes)
#         constant_values = sub_data["ConstantValues"]
#         # print("constant_values", constant_values)
#         constants = {node: constant_values[idx] for idx, node in enumerate(constant_nodes)}
#         for j, io_sub_ in tqdm(io_subs_iter, disable=not settings.progress):
#             if i == j:
#                 continue
#             sub_data_ = subs_df.iloc[j]
#             inputs_ = set(sub_data_["InputNodes"])
#             constant_nodes_ = sub_data_["ConstantNodes"]
#             # print("constant_nodes_", constant_nodes_)
#             constant_values_ = sub_data_["ConstantValues"]
#             # print("constant_values_", constant_values_)
#             constants_ = {node: constant_values_[idx] for idx, node in enumerate(constant_nodes_)}
#             if can_specialize(io_sub, io_sub_, inputs, inputs_, constants, constants_):
#                 mapping = extract_specialization_mapping(io_sub, io_sub_, inputs, inputs_, constants, constants_)
#                 spec_graph.add_edge(i, j, mapping=mapping)
#     # print("spec_graph", spec_graph)
#     graph_to_file(spec_graph, settings.out_dir / "spec_graph.dot")
#     # input(">>>>>>>>>")

# TODO: loop multiple times (tree -> MIR -> CDSL -> FLAT) not interleaved

# if len(duplicate_counts) > 0:
#     print()
#     print("Duplicates:")
#     for orig, dups in duplicate_counts.items():
#         print(f"result{orig}:\t", dups)

with MeasureTime("Finish DF", verbose=settings.times):
    stages.finalize_df(subs_df)


with MeasureTime("Apply Styles", verbose=settings.times):
    stages.apply_styles(settings, subs, io_subs, subs_df)


if settings.write.sub.enable:
    with MeasureTime("Subgraph Export", verbose=settings.times):
        stages.write_subs(settings, subs, subs_df, index_artifacts)


if settings.write.io_sub:
    with MeasureTime("Dumping I/O Subgraphs", verbose=settings.times):
        stages.write_io_subs(settings, io_subs, subs_df, index_artifacts)


if settings.write.df:
    with MeasureTime("Dump DFs", verbose=settings.times):
        stages.write_dfs(settings, subs_df, global_df, index_artifacts)


if settings.write.pie:
    with MeasureTime("Dump Pie", verbose=settings.times):
        stages.write_pie(settings, subs_df, index_artifacts)


if settings.write.sankey:
    with MeasureTime("Dump Sankey chart", verbose=settings.times):
        stages.write_sankey(settings, subs_df, index_artifacts)


if settings.write.index:
    with MeasureTime("Write Index", verbose=settings.times):
        stages.write_index(settings, GF, G, subs, io_subs, sub_stmts, subs_df, global_df, index_artifacts)


HDF5_FILE = "/tmp/mytestfile.hdf5"


print("settings.read_hdf5", settings.read_hdf5, type(settings.read_hdf5))
if settings.read_hdf5:
    with MeasureTime("HDF5 Read", verbose=settings.times):
        stages.read_hdf5(settings, subs, subs_df, HDF5_FILE)


if settings.write.hdf5:
    with MeasureTime("HDF5 Export", verbose=settings.times):
        stages.write_hdf5(settings, subs, subs_df, HDF5_FILE)


if settings.times:
    print(MeasureTime.summary())
    MeasureTime.write_csv(settings.out_dir / "times.csv", append=settings.append_times)
