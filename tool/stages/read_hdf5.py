import logging

from tqdm import tqdm

import h5py


logger = logging.getLogger("read_hdf5")


def read_hdf5(settings, subs, subs_df, file):
    logger.info("Reading HDF5...")
    with h5py.File(file, "r") as f:
        filtered_subs_df = subs_df[(subs_df["Status"] & settings.read_hdf5_flt.value) > 0].copy()
        subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
        for i, sub in tqdm(subs_iter, disable=not settings.progress):
            sub_data = subs_df.iloc[i]
            sub_hash = sub_data["SubHash"]
            io_sub_hash = sub_data["IOSubHash"]
            full_hash = sub_data["FullHash"]
            global_hash = sub_data["GlobalHash"]
            ignore_prefix = False
            if ignore_prefix:
                raise NotImplementedError
            else:
                lookup = f"{settings.session}/{settings.func}/{settings.bb}/{global_hash}/{sub_hash}/{io_sub_hash}/{full_hash}"
                # print("lookup", lookup)
                found = lookup in f
                # print("found", found)
                if found:
                    # print("f?", f[lookup], list(f[lookup]))
                    already_selected = "selected" in f[lookup]
                    print("already_selected", already_selected)
                    # input(">>>")
            # dset = f.create_dataset(dest, sub_data.to_frame().T.to_records(index=False), dtype=np.float32)
            # sub_data.to_hdf(HDF5_FILE, key=dest, mode="a")
            # sub_data.to_frame().T.to_hdf(file, key=dest, mode="a")
