import logging

from tqdm import tqdm

from ..gen.mir import generate_mir as generate_mir_

logger = logging.getLogger("generate_mir")


def generate_mir(settings, subs, GF, subs_df, filtered_subs_df, index_artifacts, sub_stmts, errs, topo):
    logger.info("Generation of MIR...")
    subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
    # for i, sub in enumerate(tqdm(subs, disable=not settings.progress)):
    for i, sub in tqdm(subs_iter, disable=not settings.progress):
        sub_data = subs_df.iloc[i]
        mir_code = generate_mir_(sub, sub_data, topo, GF, name=f"result{i}")
        with open(settings.out_dir / f"result{i}.mir", "w") as f:
            f.write(mir_code)
        index_artifacts[i]["mir"] = settings.out_dir / f"result{i}.mir"
