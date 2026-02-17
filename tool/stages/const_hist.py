import logging
from collections import defaultdict

from tqdm import tqdm
import networkx as nx
import networkx.algorithms.isomorphism as iso


logger = logging.getLogger("const_hist")


def const_hist(settings, io_subs, subs_df):
    logger.info("Creating Constants Histograms...")
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.gen_flt.value) > 0].copy()
    io_subs_iter = [(i, io_sub) for i, io_sub in enumerate(io_subs) if i in filtered_subs_df.index]
    # for i, sub in enumerate(tqdm(subs, disable=not settings.progress)):
    covered = set()
    sub_const_value_subs = {}
    for i, io_sub in tqdm(io_subs_iter, disable=not settings.progress):
        # print("i", i)
        # print("io_sub", io_sub)
        sub_data = subs_df.iloc[i]
        constants = sub_data["ConstantNodes"]
        constant_values = sub_data["ConstantValues"]
        # print("constants", constants)
        # print("constant_values", constant_values)
        const_value_subs = {}
        for c, constant in enumerate(constants):
            const_value_subs[c] = defaultdict(list)
            constant_value = constant_values[c]
            const_value_subs[c][constant_value].append(i)

        for j, io_sub_ in tqdm(io_subs_iter, disable=not settings.progress, leave=False):
            if j <= i:
                continue
            if j in covered:
                # print("skip")
                continue
            # print("j", j)
            # print("io_sub_", io_sub_)
            sub_data_ = subs_df.iloc[j]  # TODO: use loc instead!
            constants_ = sub_data_["ConstantNodes"]
            constant_values_ = sub_data_["ConstantValues"]
            # print("constants_", constants_)
            # print("constant_values_", constant_values_)

            nm = iso.categorical_node_match("hash_attr_ignore_const", None)
            # nm = iso.categorical_node_match("hash_attr", None)
            em = iso.categorical_edge_match("hash_attr", None)
            matcher = nx.algorithms.isomorphism.DiGraphMatcher(io_sub, io_sub_, node_match=nm, edge_match=em)
            check = matcher.is_isomorphic()
            # print("check", check)
            if check:
                covered.add(j)
                mapping = matcher.mapping
                # print("mapping", mapping)
                for c, constant in enumerate(constants):
                    # print("c", c)
                    # print("constant", constant)
                    constant_value = constant_values[c]
                    # print("constant_value", constant_value)
                    matching_constant = mapping[constant]
                    assert matching_constant in constants_
                    # print("matching_constant", matching_constant)
                    c_ = constants_.index(matching_constant)
                    # print("c_", c_)
                    matching_constant_value = constant_values_[c_]
                    # print("matching_constant_value", matching_constant_value)
                    const_value_subs[c][matching_constant_value].append(j)
        # print("const_value_subs", const_value_subs)
        sub_const_value_subs[i] = const_value_subs
        # TODO: implement generation
    # print("sub_const_value_subs", sub_const_value_subs)
    # input(">")
