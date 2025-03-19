import logging


from ..iso import calc_io_isos
from ..enums import ExportFilter

logger = logging.getLogger("check_iso")


def check_iso(settings, io_subs, subs_df):
    logger.info("Checking isomorphism...")
    # print("io_subs", [str(x) for x in io_subs], len(io_subs))
    # io_isos, sub_io_isos = calc_io_isos(io_subs, progress=settings.progress)
    io_isos, sub_io_isos = calc_io_isos(io_subs, progress=settings.progress, subs_df=subs_df)
    # for sub, isos in sub_io_isos.items():
    #     print("sub", sub)
    #     print("isos", isos)
    #     print("1", subs_df.loc[sub, "SubHash"], subs_df.loc[sub, "IOSubHash"])
    #     for iso in isos:
    #         print("iso", iso)
    #         print("2", subs_df.loc[iso, "SubHash"], subs_df.loc[iso, "IOSubHash"])
    #         if subs_df.loc[iso, "SubHash"] != subs_df.loc[sub, "SubHash"]:
    #             input("!1")
    #         if subs_df.loc[iso, "IOSubHash"] != subs_df.loc[sub, "IOSubHash"]:
    #             input("!2")

    # print("subs_df")
    # print(subs_df)
    # print("io_isos", io_isos, len(io_isos))
    subs_df.loc[list(io_isos), "Status"] = ExportFilter.ISO
    return io_isos, sub_io_isos
