import logging

from tqdm import tqdm


logger = logging.getLogger("apply_styles")


def apply_styles(settings, subs, io_subs, subs_df):
    logger.info("Applying styles...")
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.sub.flt) > 0].copy()
    io_subs_iter = [(i, io_sub) for i, io_sub in enumerate(io_subs) if i in filtered_subs_df.index]
    for i, io_sub in tqdm(io_subs_iter, disable=not settings.progress):
        # sub = subs[i]
        # nodes = sub.nodes
        sub_data = subs_df.iloc[i]
        # print("sub_data", sub_data)
        inputs = sub_data["InputNodes"]
        input_names = sub_data["InputNames"]
        outputs = sub_data["OutputNodes"]
        output_names = sub_data["OutputNames"]
        constants = sub_data["ConstantNodes"]
        SHOW_NODE_IDS = True
        # io_sub_topo = list(reversed(list(nx.topological_sort(io_sub))))
        # inputs_sorted = sorted(inputs, key=lambda x: io_sub_topo.index(x))
        # input_node_mapping = {n: f"src{i}" for i, n in enumerate(inputs_sorted)}
        # SHOW_NODE_IDS = False
        # for inp in inputs:
        j = 0
        for edge in io_sub.edges(data=True, keys=True):
            u, v, k, data = edge
            properties = data["properties"]
            op_idx = properties.get("op_idx")
            out_idx = properties.get("out_idx")
            edge_annotation = f"{out_idx} -> {op_idx}"
            io_sub[u][v][k]["xlabel"] = edge_annotation
        for node in io_sub.nodes:
            # print("node", node)
            # print("io_sub.nodes[node]", io_sub.nodes[node])
            if node in inputs:
                # TODO: physreg?
                io_sub.nodes[node]["xlabel"] = "IN"
                io_sub.nodes[node]["fillcolor"] = "gray"
                io_sub.nodes[node]["style"] = "filled"
                io_sub.nodes[node]["shape"] = "box"
                # label = f"src{j}"
                input_label = input_names[inputs.index(node)]
                label = input_label
                # io_sub.nodes[node]["name"] = label
                if SHOW_NODE_IDS:
                    assert "font" not in label
                    label = f'<{label}<br/><font point-size="10">{node}</font>>'
                io_sub.nodes[node]["label"] = label
                j += 1
            elif node in outputs:
                output_label = output_names[outputs.index(node)]
                io_sub.nodes[node]["xlabel"] = "OUT"
                io_sub.nodes[node]["fillcolor"] = "gray"
                io_sub.nodes[node]["style"] = "filled"
                io_sub.nodes[node]["shape"] = "box"
                # io_sub.nodes[node]["name"] = label
                if SHOW_NODE_IDS:
                    output_label = f'<{output_label}<br/><font point-size="10">{node}</font>>'
                io_sub.nodes[node]["label"] = output_label
            elif node in constants:
                if io_sub.nodes[node]["label"] == "Const":
                    label = io_sub.nodes[node]["properties"]["inst"][:-1]
                else:
                    label = io_sub.nodes[node]["label"]
                io_sub.nodes[node]["xlabel"] = "CONST"
                io_sub.nodes[node]["fillcolor"] = "lightgray"
                io_sub.nodes[node]["style"] = "filled"
                io_sub.nodes[node]["shape"] = "box"
                # io_sub.nodes[node]["name"] = label
                if SHOW_NODE_IDS:
                    assert "font" not in label
                    label = f'<{label}<br/><font point-size="10">{node}</font>>'
                io_sub.nodes[node]["label"] = label
            else:
                label = io_sub.nodes[node]["label"]
                # io_sub.nodes[node]["name"] = label
                if SHOW_NODE_IDS:
                    # assert "font" not in label
                    if "font" not in label:  # non-input nodes are shared between subs!
                        label = f'<{label}<br/><font point-size="10">{node}</font>>'
                io_sub.nodes[node]["label"] = label
            # for node in sub.nodes:
            #     label = sub.nodes[node]["label"]
            #     # sub.nodes[node]["name"] = label
            #     label = f"<{label}<br/><font point-size=\"10\">{node}</font>>"
            #     sub.nodes[node]["label"] = label
    # print("QWE")
    # input("^^^")
