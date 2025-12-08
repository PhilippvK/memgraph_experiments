import logging
import pickle
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm


from ..enums import ExportFormat
from ..graph_utils import graph_to_file

logger = logging.getLogger("write_subs")


def worker(i, sub, settings):
    # if i not in filtered_subs_df.index:
    #     continue
    ret = {}
    if settings.write.sub.fmt & ExportFormat.DOT:
        graph_to_file(sub, settings.out_dir / f"sub{i}.dot")
    if settings.write.sub.fmt & ExportFormat.PDF:
        graph_to_file(sub, settings.out_dir / f"sub{i}.pdf")
    if settings.write.sub.fmt & ExportFormat.PNG:
        graph_to_file(sub, settings.out_dir / f"sub{i}.png")
    if settings.write.sub.fmt & ExportFormat.PKL:
        with open(settings.out_dir / f"sub{i}.pkl", "wb") as f:
            pickle.dump(sub.copy(), f)
        ret["sub"] = settings.out_dir / f"sub{i}.pkl"
    return i, ret


def write_subs(settings, subs, subs_df, index_artifacts):
    logger.info("Exporting subgraphs...")
    filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.sub.flt) > 0].copy()
    subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
    with ProcessPoolExecutor(settings.n_parallel) as pool:
        futures = []
        for i, sub in subs_iter:
            future = pool.submit(worker, i, sub, settings)
            futures.append(future)
        for future in tqdm(futures, disable=not settings.progress):
            # TODO: except failing?
            i, update = future.result()
            index_artifacts[i].update(update)
