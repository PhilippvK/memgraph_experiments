import logging

from tqdm import tqdm

from ..gen.desc import generate_desc
from ..gen.flat import generate_flat_code

logger = logging.getLogger("generate_flat")


def generate_flat(settings, subs, subs_df, filtered_subs_df, index_artifacts, sub_stmts, errs):
    logger.info("Generation of FLAT...")
    subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
    for i, sub in tqdm(subs_iter, disable=not settings.progress):
        # if i not in filtered_subs_df.index:
        #     continue
        stmts = sub_stmts.get(i, None)
        assert stmts is not None, "FLAT needs TREE"
        sub_data = subs_df.iloc[i]
        try:
            desc = generate_desc(i, sub_data, name=f"result{i}")
            flat_code = generate_flat_code(stmts, desc=desc)
        except AssertionError as e:
            logger.exception(e)
            if settings.halt_on_error:
                print("FLAT ERR", str(e))
                input()
            errs.add(i)
            continue
        with open(settings.out_dir / f"result{i}.flat", "w") as f:
            f.write(flat_code)
        index_artifacts[i]["flat"] = settings.out_dir / f"result{i}.flat"
