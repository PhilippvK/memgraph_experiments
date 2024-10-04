from collections import defaultdict

import networkx as nx
import networkx.algorithms.isomorphism as iso
from tqdm import tqdm


# def calc_sub_io_isos(io_sub, io_subs):
def calc_sub_io_isos(io_sub, io_subs, i, subs_df=None, ignore_hash: bool = False):
    # TODO: label -> name
    # def node_match(x, y):
    #     # print("node_match")
    #     # # print("x", x)
    #     # print("x.label", x["label"])
    #     # # print("y", y)
    #     # print("y.label", y["label"])
    #     return (
    #         x["label"] == y["label"]
    #         and (x["label"] != "Const" or x["properties"]["inst"] == y["properties"]["inst"])
    #         and (x.get("alias", None) == y.get("alias", None))
    #         # and (x["properties"].get("alias", None) == y["properties"].get("alias", None))
    #     )

    # def edge_match(x, y):
    #     # print("edge_match")
    #     assert len(x.keys()) == 1
    #     assert len(y.keys()) == 1
    #     xkey = list(x.keys())[0]
    #     ykey = list(y.keys())[0]
    #     # print("x", x)
    #     # print("x.op_idx", x[xkey]["properties"]["op_idx"])
    #     # print("x.op_reg_single_use", x[xkey]["properties"]["op_reg_single_use"])
    #     # print("y", y)
    #     # print("y.op_idx", y[ykey]["properties"]["op_idx"])
    #     # print("y.op_reg_single_use", y[ykey]["properties"]["op_reg_single_use"])
    #     # input("o")
    #     return x[xkey]["properties"]["op_idx"] == y[ykey]["properties"]["op_idx"]

    # def node_match_(*args):
    #     print("node_match_")
    #     print("args", args)
    #     ret1 = node_match(*args)
    #     ret2 = iso.categorical_node_match("hash_attr", None)(*args)
    #     print("ret1", ret1)
    #     print("ret2", ret2)
    #     input(">")
    #     return ret1

    # def edge_match_(*args):
    #     print("edge_match_")
    #     print("args", args)
    #     ret1 = edge_match(*args)
    #     ret2 = iso.categorical_edge_match("hash_attr", None)(*args)
    #     print("ret1", ret1)
    #     print("ret2", ret2)
    #     input(">")
    #     return ret1

    def helper(io_sub, io_sub_, i, j):
        # print("helper", i, j)
        # print("helper")
        # print("io_sub", io_sub)
        # print("io_sub_", io_sub_)
        # ret = nx.is_isomorphic(io_sub, io_sub_, node_match=node_match, edge_match=edge_match)
        if subs_df is not None and not ignore_hash:
            # TODO: iloc vs loc?
            a = subs_df.loc[i, "IOSubHash"]
            # print("a", a)
            b = subs_df.loc[j, "IOSubHash"]
            # print("b", b)
            ret = a == b
            # print("ret", ret)
            if ret:
                ret = nx.is_isomorphic(
                    io_sub,
                    io_sub_,
                    node_match=iso.categorical_node_match("hash_attr", None),
                    edge_match=iso.categorical_edge_match("hash_attr", None),
                    # node_match=node_match_,
                    # edge_match=edge_match_,
                )
                # print("ret2", ret2)
                # if ret != ret2:
                #     input("<>")
            else:
                pass
        else:
            ret = nx.is_isomorphic(
                io_sub,
                io_sub_,
                node_match=iso.categorical_node_match("hash_attr", None),
                edge_match=iso.categorical_edge_match("hash_attr", None),
                # node_match=node_match_,
                # edge_match=edge_match_,
            )
        # if i == 805 and j == 807:
        # print("ret", ret)
        # input("?")
        return ret

    # def helper2(args):
    #     i, j, io_sub, io_sub_ = args
    #     # print("helper", i, j)
    #     ret = nx.is_isomorphic(io_sub, io_sub_, node_match=node_match)
    #     # if i == 805 and j == 807:
    #     #     print("ret", ret)
    #     #     input("?")
    #     return ret

    # def helper3(args):
    #     ret = []
    #     for i, j, io_sub, io_sub_ in args:
    #         # print("helper", i, j)
    #         ret_ = nx.is_isomorphic(io_sub, io_sub_, node_match=node_match)
    #         ret.append(ret_)
    #         # if i == 805 and j == 807:
    #         #     print("ret", ret)
    #         #     input("?")
    #     return ret

    # A:
    # io_isos_ = set(
    #     j for j, io_sub_ in enumerate(io_subs) if j > i and j not in io_isos and helper(i, j, io_sub, io_sub_)
    # )
    # B:
    io_isos_ = set(j for j, io_sub_ in io_subs if helper(io_sub, io_sub_, i, j))
    # C:
    # to_check = [(i, j, io_sub, io_sub_) for j, io_sub_ in enumerate(io_subs) if j > i and j not in io_isos]
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     res = executor.map(helper2, to_check)
    #     io_isos_ = set(to_check[k][1] for k, is_iso in enumerate(res) if is_iso)
    # D:
    # to_check = [(i, j, io_sub, io_sub_) for j, io_sub_ in enumerate(io_subs) if j > i and j not in io_isos]
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     res = executor.map(helper2, to_check)
    #     io_isos_ = set(to_check[k][1] for k, is_iso in enumerate(res) if is_iso)
    # E:
    # to_check = [(i, j, io_sub, io_sub_) for j, io_sub_ in enumerate(io_subs) if j > i and j not in io_isos]

    # def batch(iterable, n=1):
    #     l = len(iterable)
    #     for ndx in range(0, l, n):
    #         yield iterable[ndx : min(ndx + n, l)]

    # batch_size = len(to_check) // 36
    # to_check_batches = list(batch(to_check, n=batch_size))
    # # with concurrent.futures.ProcessPoolExecutor() as executor:
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     res_batched = executor.map(helper3, to_check_batches)
    #     # print("res_batched", res_batched)
    #     res = sum(res_batched, [])
    #     # print("res", res)
    #     io_isos_ = set(to_check[k][1] for k, is_iso in enumerate(res) if is_iso)
    return io_isos_


def calc_io_isos(io_subs, progress: bool = False, subs_df=None, ignore_hash: bool = False):
    io_isos = set()
    sub_io_isos = defaultdict(list)
    for i, io_sub in enumerate(tqdm(io_subs, disable=not progress)):
        # break  # TODO
        # print("io_sub", i, io_sub, io_sub.nodes)
        # print("io_sub nodes", [GF.nodes[n] for n in io_sub.nodes])
        if i in io_isos:
            continue
        to_check = [(j, io_sub_) for j, io_sub_ in enumerate(io_subs) if j > i and j not in io_isos]
        io_isos_ = calc_sub_io_isos(io_sub, to_check, i, subs_df=subs_df, ignore_hash=ignore_hash)

        sub_io_isos[i] += list(io_isos_)
        # print("io_isos_", io_isos_)
        # io_iso_count = len(io_isos_)
        # print("io_iso_count", io_iso_count)
        io_isos |= io_isos_
    return io_isos, sub_io_isos
