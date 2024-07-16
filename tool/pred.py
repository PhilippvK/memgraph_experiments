from .enums import InstrPredicate


def detect_predicates(sub):
    sub_predicates = InstrPredicate.NONE
    print("check_predicates", sub)
    for node in sub.nodes:
        node_data = sub.nodes[node]
        name = node_data.get("label", "?")
        print("name", name)
        # TODO: do not hardcode here (rather add to memgraph db)
        LOADS = ["LD", "LW", "LWU", "LH", "LHU", "LB", "LBU"]
        STORES = ["SD", "SW", "SH", "SB"]
        BRANCHES = ["BEQ", "BNE"]
        if name in LOADS:
            sub_predicates |= InstrPredicate.MAY_LOAD
        if name in STORES:
            sub_predicates |= InstrPredicate.MAY_STORE
        if name in BRANCHES:
            sub_predicates |= InstrPredicate.IS_BRANCH
    return sub_predicates


def check_predicates(pred, allowed_predicates):
    ret = (pred & allowed_predicates) == pred
    print("ret", ret)
    return ret
