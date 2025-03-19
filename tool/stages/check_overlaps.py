import logging

from ..graph_utils import calc_weights_iso

logger = logging.getLogger("check_overlaps")


def check_overlaps(subs, GF, subs_df, io_isos, sub_io_isos):
    logger.info("Checking overlaps...")
    for key in sub_io_isos:
        iso_nodes = set()
        iso_nodes |= set(subs[key].nodes)
        val = sub_io_isos[key]
        num_isos = len(val)
        non_overlapping = set()

        def check_overlap(x, y):
            # print("check_overlap", x, y)
            intersection = set(x.nodes) & set(y.nodes)
            # print("intersection", intersection)
            return len(intersection) > 0

        for iso in val:
            iso_nodes |= set(subs[iso].nodes)
            # print("iso", iso)
            ol = False
            ol |= check_overlap(subs[iso], subs[key])
            if not ol:
                for iso_ in non_overlapping:
                    # print("iso_", iso_)
                    ol |= check_overlap(subs[iso], subs[key])
                    if ol:
                        break
            if not ol:
                non_overlapping.add(iso)

        # print("num_isos", num_isos)
        # print("non_overlapping", non_overlapping)
        # print("len(non_overlapping)", len(non_overlapping))
        # input("LLL")
        subs_df.at[key, "IsoNodes"] = list(iso_nodes)
        iso_weight, _ = calc_weights_iso(GF, iso_nodes)
        subs_df.loc[key, "IsoWeight"] = iso_weight
        subs_df.loc[key, "#IsosNO"] = len(non_overlapping)
        subs_df.at[key, "IsosNO"] = list(non_overlapping)
        subs_df.loc[key, "#Isos"] = num_isos
        subs_df.at[key, "Isos"] = list(set(val))
        if num_isos == 0:
            continue
        assert key not in io_isos
        # print(f"{key}: {num_isos}")
