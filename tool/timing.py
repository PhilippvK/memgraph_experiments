import logging
from typing import Union
from pathlib import Path
from time import perf_counter
from contextlib import ContextDecorator

import pandas as pd


logger = logging.getLogger("timing")


class MeasureTime(ContextDecorator):
    times = {}

    @staticmethod
    def reset():
        MeasureTime.times = {}

    @staticmethod
    def write_csv(dest: Union[str, Path]):
        df = pd.DataFrame.from_dict(MeasureTime.times, orient="index", columns=["Secs"])
        df.index.name = "Name"
        df["Secs (rel.)"] = df["Secs"] / df["Secs"].sum()
        df.to_csv(dest)

    @staticmethod
    def summary(rel: bool = True, sort: bool = False):
        if sort:
            raise NotImplementedError

        def helper(msg, elapsed, total, rel):
            ret = f"- {msg}: {elapsed:.3f} sec"
            if rel:
                ret += f" ({elapsed*100/total:.1f}%)"
            return ret

        total = sum(MeasureTime.times.values())
        return "Execution Time Summary:\n" + "\n".join(
            [helper(msg, elapsed, total, rel) for msg, elapsed in MeasureTime.times.items()]
        )

    def __init__(self, msg: str, verbose: bool = True):
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        self.time = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        elapsed = perf_counter() - self.time
        assert self.msg not in MeasureTime.times
        MeasureTime.times[self.msg] = elapsed
        if self.verbose:
            log_func = logger.info if self.verbose else logger.debug
            log_func(f"<{self.msg}> took {elapsed:.3f} seconds")
