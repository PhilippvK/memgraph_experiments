def add_hash_attr(sub, attr_name: str = "hash_attr", ignore_const: bool = False):
    for node in sub.nodes:
        temp = sub.nodes[node]["label"]
        if temp == "Const" and not ignore_const:
            temp += "-" + sub.nodes[node]["properties"]["inst"]
        temp += "-" + str(sub.nodes[node].get("alias", None))
        sub.nodes[node][attr_name] = temp
    for edge in sub.edges:
        sub.edges[edge][attr_name] = str(sub.edges[edge]["properties"]["op_idx"])
