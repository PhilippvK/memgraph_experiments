import logging
import pickle
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm

from ..enums import ExportFormat
from ..graph_utils import graph_to_file

logger = logging.getLogger("write_io_subs")


def worker(i, io_sub, settings):
    ret = {}
    if settings.write.io_sub_fmt & ExportFormat.DOT:
        graph_to_file(io_sub, settings.out_dir / f"io_sub{i}.dot")
    if settings.write.io_sub_fmt & ExportFormat.PDF:
        graph_to_file(io_sub, settings.out_dir / f"io_sub{i}.pdf")
    if settings.write.io_sub_fmt & ExportFormat.PNG:
        graph_to_file(io_sub, settings.out_dir / f"io_sub{i}.png")
    if settings.write.io_sub_fmt & ExportFormat.PKL:
        with open(settings.out_dir / f"io_sub{i}.pkl", "wb") as f:
            pickle.dump(io_sub.copy(), f)
        ret["io_sub"] = settings.out_dir / f"io_sub{i}.pkl"
    return i, ret


def write_io_subs(settings, io_subs, subs_df, index_artifacts):
    logger.info("Exporting I/O subgraphs...")
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.io_sub_flt) > 0].copy()
    io_subs_iter = [(i, io_sub) for i, io_sub in enumerate(io_subs) if i in filtered_subs_df.index]
    # for i, io_sub in enumerate(tqdm(io_subs, disable=not settings.progress)):
    with ProcessPoolExecutor(settings.n_parallel) as pool:
        futures = []
        for i, io_sub in io_subs_iter:
            future = pool.submit(worker, i, io_sub, settings)
            futures.append(future)
        for future in tqdm(futures, disable=not settings.progress):
            # TODO: except failing?
            i, update = future.result()
            index_artifacts[i].update(update)
