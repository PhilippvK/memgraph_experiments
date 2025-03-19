import logging

from tqdm import tqdm


logger = logging.getLogger("apply_styles")


def apply_styles(settings, subs, io_subs, subs_df):
    logger.info("Applying styles...")
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.sub.flt) > 0].copy()
    io_subs_iter = [(i, io_sub) for i, io_sub in enumerate(io_subs) if i in filtered_subs_df.index]
    for i, io_sub in tqdm(io_subs_iter, disable=not settings.progress):
        sub_data = subs_df.iloc[i]
        inputs = sub_data["InputNodes"]
        input_names = sub_data["InputNames"]
        outputs = sub_data["OutputNodes"]
        output_names = sub_data["OutputNames"]
        constants = sub_data["ConstantNodes"]
        SHOW_NODE_IDS = True
        j = 0
        for edge in io_sub.edges(data=True, keys=True):
            u, v, k, data = edge
            properties = data["properties"]
            op_idx = properties.get("op_idx")
            out_idx = properties.get("out_idx")
            edge_annotation = f"{out_idx} -> {op_idx}"
            io_sub[u][v][k]["xlabel"] = edge_annotation
        for node in io_sub.nodes:
            if node in inputs:
                # TODO: physreg?
                io_sub.nodes[node]["xlabel"] = "IN"
                io_sub.nodes[node]["fillcolor"] = "gray"
                io_sub.nodes[node]["style"] = "filled"
                io_sub.nodes[node]["shape"] = "box"
                input_label = input_names[inputs.index(node)]
                label = input_label
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
                if SHOW_NODE_IDS:
                    assert "font" not in label
                    label = f'<{label}<br/><font point-size="10">{node}</font>>'
                io_sub.nodes[node]["label"] = label
            else:
                label = io_sub.nodes[node]["label"]
                if SHOW_NODE_IDS:
                    if "font" not in label:  # non-input nodes are shared between subs!
                        label = f'<{label}<br/><font point-size="10">{node}</font>>'
                io_sub.nodes[node]["label"] = label


# TODO: move to index.py?
from ..enum import ExportFilter


def filter_candidates(candidates, flt: ExportFilter):
    return [candidate for candidate in candidates if candidate.properties["Status"] & flt]


def apply_styles_new(settings, index):
    logger.info("Applying styles...")
    candidates = index.candidates
    filtered_candidates = filter_candidates(candidates, settings.write.sub.flt)
    for candidate in tqdm(filtered_candidates, disable=not settings.progress):
        # sub = candidate.sub
        io_sub = candidate.io_sub
        sub_data = candidate.properties
        inputs = sub_data["InputNodes"]
        input_names = sub_data["InputNames"]
        outputs = sub_data["OutputNodes"]
        output_names = sub_data["OutputNames"]
        constants = sub_data["ConstantNodes"]
        SHOW_NODE_IDS = True
        j = 0
        for edge in io_sub.edges(data=True, keys=True):
            u, v, k, data = edge
            properties = data["properties"]
            op_idx = properties.get("op_idx")
            out_idx = properties.get("out_idx")
            edge_annotation = f"{out_idx} -> {op_idx}"
            io_sub[u][v][k]["xlabel"] = edge_annotation
        for node in io_sub.nodes:
            if node in inputs:
                # TODO: physreg?
                io_sub.nodes[node]["xlabel"] = "IN"
                io_sub.nodes[node]["fillcolor"] = "gray"
                io_sub.nodes[node]["style"] = "filled"
                io_sub.nodes[node]["shape"] = "box"
                input_label = input_names[inputs.index(node)]
                label = input_label
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
                if SHOW_NODE_IDS:
                    assert "font" not in label
                    label = f'<{label}<br/><font point-size="10">{node}</font>>'
                io_sub.nodes[node]["label"] = label
            else:
                label = io_sub.nodes[node]["label"]
                if SHOW_NODE_IDS:
                    if "font" not in label:  # non-input nodes are shared between subs!
                        label = f'<{label}<br/><font point-size="10">{node}</font>>'
                io_sub.nodes[node]["label"] = label
