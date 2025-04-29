from collections import defaultdict

import networkx as nx
import networkx.algorithms.isomorphism as iso
from tqdm import tqdm


def categorical_edge_match_multidigraph(attr, default):
    if isinstance(attr, str):

        def match(data1, data2):
            # print("data1", data1)
            # print("data2", data2)
            # print("data1!", {k: v.get(attr, default) for k, v in data1.items()})
            # print("data2!", {k: v.get(attr, default) for k, v in data2.items()})
            if len(data1) == 1:
                assert len(data2) == 1
                data1 = list(data1.values())[0]
                data2 = list(data2.values())[0]
                return data1.get(attr, default) == data2.get(attr, default)
            else:
                assert len(data2) == len(data1)
                temp1 = {k: v.get(attr, default) for k, v in data1.items()}
                temp2 = {k: v.get(attr, default) for k, v in data2.items()}
                vals1 = list(temp1.values())
                vals2 = list(temp2.values())
                unique_vals1 = set(vals1)
                unique_vals2 = set(vals2)
                # print("vals1", vals1)
                # print("vals2", vals2)
                # print("unique_vals1", unique_vals1)
                # print("unique_vals2", unique_vals2)
                if len(unique_vals1) != len(unique_vals2):
                    return False
                return unique_vals1 == unique_vals2

    else:
        attrs = list(zip(attr, default))  # Python 3

        def match(data1, data2):
            return all(data1.get(attr, d) == data2.get(attr, d) for attr, d in attrs)

    return match


# def calc_sub_io_isos(io_sub, io_subs):
def calc_sub_io_isos(io_sub, io_subs, i, subs_df=None, ignore_hash: bool = False, hash_attr: str = "hash_attr"):
    print("calc_sub_io_isos", io_sub, io_subs, i, subs_df is not None, ignore_hash, hash_attr)
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
                try:
                    ret = nx.is_isomorphic(
                        io_sub,
                        io_sub_,
                        node_match=iso.categorical_node_match(hash_attr, None),
                        # edge_match=iso.categorical_edge_match("hash_attr", None),
                        edge_match=categorical_edge_match_multidigraph(hash_attr, None),
                        # node_match=node_match_,
                        # edge_match=edge_match_,
                    )
                except AssertionError:
                    print("io_sub", io_sub)
                    print("io_sub_", io_sub_)
                    from networkx.drawing.nx_agraph import write_dot
                    for edge in io_sub.edges(data=True, keys=True):
                        u, v, k, data = edge
                        properties = data["properties"]
                        op_idx = properties.get("op_idx")
                        out_idx = properties.get("out_idx")
                        edge_annotation = f"{out_idx} -> {op_idx}"
                        io_sub[u][v][k]["xlabel"] = edge_annotation
                    for edge in io_sub_.edges(data=True, keys=True):
                        u, v, k, data = edge
                        properties = data["properties"]
                        op_idx = properties.get("op_idx")
                        out_idx = properties.get("out_idx")
                        edge_annotation = f"{out_idx} -> {op_idx}"
                        io_sub_[u][v][k]["xlabel"] = edge_annotation
                    import pickle
                    with open("/tmp/io_sub.pkl", "wb") as f:
                        pickle.dump(io_sub, f)
                    with open("/tmp/io_sub_.pkl", "wb") as f:
                        pickle.dump(io_sub_, f)
                    write_dot(io_sub, "/tmp/io_sub.dot")
                    write_dot(io_sub_, "/tmp/io_sub_.dot")
                    input("ASSERTION")
                # print("ret2", ret2)
                # if ret != ret2:
                #     input("<>")
            else:
                pass
        else:
            ret = nx.is_isomorphic(
                io_sub,
                io_sub_,
                node_match=iso.categorical_node_match(hash_attr, None),
                # edge_match=iso.categorical_edge_match("hash_attr", None),
                edge_match=categorical_edge_match_multidigraph(hash_attr, None),
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
    groups = defaultdict(list)
    for i, io_sub in enumerate(io_subs):
        # print("i", i)
        # print("io_sub", io_sub)
        # print("io_sub.nodes", io_sub.nodes)
        # print("len(io_sub.nodes)", len(io_sub.nodes))
        num_nodes = len(io_sub.nodes)
        groups[num_nodes].append((i, io_sub))
    # print("groups", groups)
    all_io_isos = set()
    all_sub_io_isos = defaultdict(list)
    for group_num_nodes, group_io_subs in groups.items():
        io_isos = set()
        sub_io_isos = defaultdict(list)
        for i, io_sub in tqdm(group_io_subs, disable=not progress, leave=False):
            # break  # TODO
            # print("io_sub", i, io_sub, io_sub.nodes)
            # print("io_sub nodes", [GF.nodes[n] for n in io_sub.nodes])
            if i in io_isos:
                continue
            to_check = [(j, io_sub_) for j, io_sub_ in group_io_subs if j > i and j not in io_isos]
            io_isos_ = calc_sub_io_isos(io_sub, to_check, i, subs_df=subs_df, ignore_hash=ignore_hash)

            sub_io_isos[i] += list(io_isos_)
            # print("io_isos_", io_isos_)
            # io_iso_count = len(io_isos_)
            # print("io_iso_count", io_iso_count)
            io_isos |= io_isos_
        all_io_isos |= io_isos
        all_sub_io_isos.update(sub_io_isos)
    return all_io_isos, all_sub_io_isos
