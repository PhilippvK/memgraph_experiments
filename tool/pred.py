from .enums import InstrPredicate


# def detect_predicates(sub):
#     sub_predicates = InstrPredicate.NONE
#     # print("check_predicates", sub)
#     for node in sub.nodes:
#         node_data = sub.nodes[node]
#         name = node_data.get("label", "?")
#         # print("name", name)
#         # TODO: do not hardcode here (rather add to memgraph db)
#         LOADS = ["LD", "LW", "LWU", "LH", "LHU", "LB", "LBU", "FLW", "FLD"]
#         STORES = ["SD", "SW", "SH", "SB", "FSW", "FSD"]
#         BRANCHES = ["BEQ", "BNE"]
#         if name in LOADS:
#             sub_predicates |= InstrPredicate.MAY_LOAD
#         if name in STORES:
#             sub_predicates |= InstrPredicate.MAY_STORE
#         if name in BRANCHES:
#             sub_predicates |= InstrPredicate.IS_BRANCH
#     return sub_predicates


def detect_predicates(sub):
    sub_predicates = InstrPredicate.NONE
    loads = set()
    stores = set()
    terminators = set()
    branches = set()
    # print("check_predicates", sub)
    for node in sub.nodes:
        node_data = sub.nodes[node]
        properties = node_data["properties"]
        # name = node_data.get("label", "?")
        # print("name", name)
        if properties.get("mayLoad", False):
            sub_predicates |= InstrPredicate.MAY_LOAD
            loads.add(node)
        if properties.get("mayStore", False):
            sub_predicates |= InstrPredicate.MAY_STORE
            stores.add(node)
        if properties.get("isPseudo", False):
            sub_predicates |= InstrPredicate.IS_PSEUDO
        if properties.get("isReturn", False):
            sub_predicates |= InstrPredicate.IS_RETURN
        if properties.get("isCall", False):
            sub_predicates |= InstrPredicate.IS_CALL
        if properties.get("isTerminator", False):
            sub_predicates |= InstrPredicate.IS_TERMINATOR
            terminators.add(node)
        if properties.get("isBranch", False):
            sub_predicates |= InstrPredicate.IS_BRANCH
            branches.add(node)
        if properties.get("hasUnmodeledSideEffects", False):
            sub_predicates |= InstrPredicate.HAS_UNMODELED_SIDE_EFFECTS
        if properties.get("isCommutable", False):  # TODO: This only makes sense per node?
            sub_predicates |= InstrPredicate.IS_COMMUTABLE
    return (
        sub_predicates,
        len(loads),
        loads,
        len(stores),
        stores,
        len(terminators),
        terminators,
        len(branches),
        branches,
    )


def check_predicates(pred, allowed_predicates):
    ret = (pred & allowed_predicates) == pred
    # print("ret", ret)
    return ret
