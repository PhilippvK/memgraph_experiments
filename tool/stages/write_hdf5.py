import logging

from tqdm import tqdm
# import h5py
# import numpy as np


from ..enums import ExportFilter

logger = logging.getLogger("write_hdf5")


def write_hdf5(settings, subs, subs_df):
    logger.info("Exporting HDF5...")
    # with h5py.File("/tmp/mytestfile.hdf5", "w") as f:
    if True:
        filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.hdf5_flt) > 0].copy()
        subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
        for i, sub in tqdm(subs_iter, disable=not settings.progress):
            sub_data = subs_df.iloc[i]
            # print("sub_data", sub_data.to_frame().T)
            # print("sub_data", sub_data.to_frame().T.to_records(index=False))
            # print("sub_data.dtypes", sub_data.to_frame().T.dtypes)
            sub_hash = sub_data["SubHash"]
            io_sub_hash = sub_data["IOSubHash"]
            full_hash = sub_data["FullHash"]
            global_hash = sub_data["GlobalHash"]
            status = "???"
            if sub_data["Status"] & ExportFilter.SELECTED:
                status = "selected"
            elif sub_data["Status"] & (
                ExportFilter.FILTERED_IO
                | ExportFilter.FILTERED_COMPLEX
                | ExportFilter.FILTERED_SIMPLE
                | ExportFilter.FILTERED_PRED
                | ExportFilter.FILTERED_MEM
                | ExportFilter.FILTERED_BRANCH
                | ExportFilter.FILTERED_WEIGHTS
                | ExportFilter.FILTERED_ENC
            ):
                status = "filtered"
            elif sub_data["Status"] & ExportFilter.INVALID:
                status = "invalid"
            elif sub_data["Status"] & ExportFilter.ERROR:
                status = "error"
            elif sub_data["Status"] & ExportFilter.ISO:
                status = "iso"
            # ExFilter.FILTERED_ISO
            if full_hash is None:
                # print("i", i)
                # print("sub", sub)
                # print("sub_data", sub_data)
                input("why?")
            dt = sub_data["DateTime"]
            # dest = f"{sub_hash}/{io_sub_hash}/{full_hash}/{global_hash}/{dt}"
            dest = f"{settings.session}/{settings.func}/{settings.bb}/{global_hash}/{sub_hash}/{io_sub_hash}/{full_hash}/{status}/{dt}"
            # print("dest", dest)
            # dset = f.create_dataset(dest, sub_data.to_frame().T.to_records(index=False), dtype=np.float32)
            sub_data.to_hdf("/tmp/mytestfile.hdf5", key=dest, mode="a")
            # sub_data.to_frame().T.to_hdf("/tmp/mytestfile.hdf5", key=dest, mode="a")
