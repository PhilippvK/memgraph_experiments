import logging

import networkx as nx

logger = logging.getLogger("relabel_nodes")


def relabel_nodes(settings, G, GF, subs):
    logger.info("Relabeling nodes...")
    # print("GF", GF)
    # print("GF.nodes", GF.nodes)
    mapping = dict(zip(GF.nodes.keys(), range(len(GF.nodes))))
    # mapping1 = mapping
    # print("subs[0].nodes", subs[0].nodes)
    GF = nx.relabel_nodes(GF, mapping)
    G = nx.relabel_nodes(G, mapping)
    for i in range(len(subs)):
        subs[i] = nx.relabel_nodes(subs[i], mapping)
    # print("subs[0].nodes", subs[0].nodes)
    # input("1!")
    # print("GF", GF)
    # print("G", G)
    # print("GF.nodes", GF.nodes)
    # print("G.nodes", G.nodes)
    # mapping = dict(zip(G.nodes.keys(), range(len(G.nodes))))
    # G = nx.relabel_nodes(G, mapping)
    # for i in range(len(subs)):
    #     subs[i] = nx.relabel_nodes(subs[i], mapping)
    # print("G", G)
    # print("G.nodes", G.nodes)
    # topo = list(reversed(list(nx.topological_sort(G))))
    topo = list(reversed(list(nx.topological_sort(GF))))
    # print("subs[0].nodes", subs[0].nodes)
    # print("topo", topo)
    # mapping = dict(zip(G.nodes.keys(), topo))
    # mapping = dict(zip(GF.nodes.keys(), topo))
    mapping = dict(zip(topo, list(range(len(GF.nodes)))))
    # print("mapping", mapping)
    GF = nx.relabel_nodes(GF, mapping)
    G = nx.relabel_nodes(G, mapping)
    for i in range(len(subs)):
        subs[i] = nx.relabel_nodes(subs[i], mapping)
    # print("subs[0].nodes", subs[0].nodes)
    topo = list(nx.topological_sort(GF))
    # print("topo", topo)
    # print("topo for subs[0]", list(nx.topological_sort(subs[0])))
    # input("2!")
    # print("GF", GF)
    # print("G", G)
    # print("GF.nodes", GF.nodes)
    # print("G.nodes", G.nodes)
    # topo2 = list(reversed(list(nx.topological_sort(GF))))
    # print("topo2", topo2)

    # # @A
    # topo2 = list(nx.topological_sort(subs[380]))
    # print("topo2", topo2)
    # # ADDI
    # print("mapping1[260397]", mapping1[260397])
    # print("mapping[1094]", mapping[1094])
    # print("55", subs[380].nodes[55])
    # # BEQ
    # print("mapping1[260396]", mapping1[260396])
    # print("mapping[1093]", mapping[1093])
    # print("57", subs[380].nodes[57])
    #
    # # @B
    # topo2 = list(nx.topological_sort(subs[383]))
    # print("topo2", topo2)
    # # SLTU
    # print("mapping1[260420]", mapping1[260420])
    # print("mapping[1116]", mapping[1116])
    # print("6", subs[383].nodes[6])
    # # SLLI
    # print("mapping1[260421]", mapping1[260421])
    # print("mapping[1117]", mapping[1117])
    # print("4", subs[383].nodes[4])
    topo = list(nx.topological_sort(GF))

    # print("subs[0]", subs[0])
    # print("subs[0].nodes", subs[0].nodes)
