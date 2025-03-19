import json
from hashlib import sha1
import numpy as np


def add_hash_attr(sub, attr_name: str = "hash_attr", ignore_const: bool = False, ignore_names: bool = False):
    for node in sub.nodes:
        print("node", node)
        print("sub.nodes[node]", sub.nodes[node])
        temp = sub.nodes[node]["label"]

        def drop_style(label):
            print("drop_style", label)
            label = label[1:]
            if "<br/>" in label:
                label = label.split("<br/>", 1)[0]
            # label = label.replace("<br/>", "")
            # label = label.replace("</font>", "")
            # label = label.replace("<font point-size=\"10\">", "")  # TODO: use re
            return label
        if temp[0] == "<":
            temp = drop_style(temp)
        if ignore_names and temp.startswith("src"):
            temp = "src"
        if temp == "Const" and not ignore_const:
            temp += "-" + sub.nodes[node]["properties"]["inst"]
        temp += "-" + str(sub.nodes[node].get("alias", None))
        print("temp", temp)
        sub.nodes[node][attr_name] = temp
    for edge in sub.edges:
        sub.edges[edge][attr_name] = str(sub.edges[edge]["properties"]["op_idx"])


def check_type(x):
    if isinstance(x, set):
        x = sorted(list(x))
    elif isinstance(x, np.ndarray):
        x = x.tolist()
    elif isinstance(x, float):
        if np.isnan(x):
            x = None
    elif isinstance(x, np.int64):
        x = float(x)
    assert x is None or isinstance(x, (str, int, list, float)), f"Unsupported: {type(x)}"
    return x


def calc_full_hash(sub_data):
    data = {}
    data["IOSubHash"] = sub_data["IOSubHash"]
    data["InputNames"] = sub_data["InputNames"]
    data["ConstantValues"] = sub_data["ConstantValues"]
    data["OutputNames"] = sub_data["OutputNames"]
    data["OperandNames"] = sub_data["OperandNames"]
    data["OperandDirs"] = sub_data["OperandDirs"]
    data["OperandTypes"] = sub_data["OperandTypes"]
    data["OperandRegClasses"] = sub_data["OperandRegClasses"]
    data["OperandEncBits"] = sub_data["OperandEncBits"]
    data["Constraints"] = sub_data["Constraints"]
    data["Variations"] = sub_data["Variations"]
    data["Instrs"] = sub_data["Instrs"]

    data = {k: check_type(v) for k, v in data.items()}

    ret = json.dumps(data, sort_keys=True)

    ret2 = sha1(ret.encode("utf-8")).hexdigest()
    return ret2


def calc_global_hash(global_df):
    pass
    # TODO:
    # ["session", "func", "bb"]

    assert len(global_df) == 1
    data = global_df[
        [
            "min_inputs",
            "max_inputs",
            "min_outputs",
            "max_outputs",
            "min_nodes",
            "max_nodes",
            "xlen",
            "stage",
            "limit_results",
            "min_path_len",
            "max_path_len",
            "max_path_width",
            "instr_predicates",
            "ignore_names",
            "ignore_op_types",
            "allowed_enc_sizes",
            "max_enc_footprint",
            "max_enc_weight",
            "min_enc_bits_left",
            "min_iso_weight",
            "max_loads",
            "max_stores",
            "max_mems",
            "max_branches",
        ]
    ].to_dict("records")[0]
    data = {k: check_type(v) for k, v in data.items()}

    ret = json.dumps(data, sort_keys=True)

    ret2 = sha1(ret.encode("utf-8")).hexdigest()
    return ret2
