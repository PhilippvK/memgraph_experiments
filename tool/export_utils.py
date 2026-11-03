"""Scheduling shared by the two candidate graph exporters."""

import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

from tqdm import tqdm

from .enums import ExportFormat


def export_graphs(worker, items, settings, formats, index_artifacts):
    """Export independent graphs and merge artifact updates in the parent.

    Process startup outweighs export time for small batches and pickle-only
    output. Rendering larger batches retains the existing process parallelism.
    """
    requested = settings.n_parallel
    if requested is not None and (
        isinstance(requested, bool) or not isinstance(requested, int) or requested < 1
    ):
        raise ValueError("n_parallel must be a positive integer or null")
    graph_formats = ExportFormat.DOT | ExportFormat.PDF | ExportFormat.PNG | ExportFormat.PKL
    if not items or not formats & graph_formats:
        return
    workers = min(requested or os.cpu_count() or 1, len(items))
    serial = workers == 1 or len(items) <= 8 or not formats & (
        ExportFormat.DOT | ExportFormat.PDF | ExportFormat.PNG
    )

    def collect(results):
        for i, update in tqdm(results, total=len(items), disable=not settings.progress):
            index_artifacts[i].update(update)

    if serial:
        collect(worker(i, graph, settings) for i, graph in items)
    else:
        with ProcessPoolExecutor(workers) as pool:
            ids, graphs = zip(*items)
            collect(pool.map(worker, ids, graphs, repeat(settings)))
