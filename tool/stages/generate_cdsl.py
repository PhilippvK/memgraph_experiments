import logging

from tqdm import tqdm

from ..gen.desc import generate_desc
from ..gen.cdsl import generate_cdsl as generate_cdsl_

logger = logging.getLogger("generate_cdsl")


def generate_cdsl(settings, subs, subs_df, filtered_subs_df, index_artifacts, sub_stmts, errs):
    logger.info("Generation of CDSL...")
    subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
    # for i, sub in enumerate(tqdm(subs, disable=not settings.progress)):
    for i, sub in tqdm(subs_iter, disable=not settings.progress):
        sub_data = subs_df.iloc[i]
        stmts = sub_stmts.get(i, None)
        assert stmts is not None, "CDSL needs TREE"
        try:
            desc = generate_desc(i, sub_data, name=f"result{i}")
            cdsl_code = generate_cdsl_(
                stmts,
                sub_data,
                xlen=settings.riscv.xlen,
                name=f"result{i}",
                desc=desc,
            )
        except AssertionError as e:
            logger.exception(e)
            if settings.halt_on_error:
                print("CDSL ERR", str(e))
                input()
            errs.add(i)
            continue
        with open(settings.out_dir / f"result{i}.core_desc", "w") as f:
            f.write(cdsl_code)
        if index_artifacts is not None:
            index_artifacts[i]["cdsl"] = settings.out_dir / f"result{i}.core_desc"
