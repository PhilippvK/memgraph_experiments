import logging
import pickle

from tqdm import tqdm
from anytree import RenderTree
from anytree.dotexport import RenderTreeGraph

from ..gen.desc import generate_desc
from ..gen.tree import generate_tree as generate_tree_
from ..enums import ExportFormat

logger = logging.getLogger("generate_tree")


def generate_tree(settings, subs, io_subs, subs_df, filtered_subs_df, index_artifacts, sub_stmts, errs):
    logger.info("Generation of Tree...")
    subs_iter = [(i, sub) for i, sub in enumerate(subs) if i in filtered_subs_df.index]
    # for i, sub in enumerate(tqdm(subs, disable=not settings.progress)):
    for i, sub in tqdm(subs_iter, disable=not settings.progress):
        # print("i", i)
        # print("sub", sub)
        sub_data = subs_df.iloc[i]
        io_sub = io_subs[i]
        try:
            stmts = generate_tree_(
                sub,
                sub_data,
                # GF,
                io_sub,
                xlen=settings.riscv.xlen,
            )
            # input(">>>")
            sub_stmts[i] = stmts
        except AssertionError as e:
            logger.exception(e)
            if settings.halt_on_error:
                print("TREE ERR", str(e))
                input()
            errs.add(i)
            continue
        if settings.write.tree_fmt & ExportFormat.PKL:
            with open(settings.out_dir / f"tree{i}.pkl", "wb") as f:
                pickle.dump(stmts, f)
            index_artifacts[i]["tree"] = settings.out_dir / f"tree{i}.pkl"
        if settings.write.tree_fmt & ExportFormat.TXT:
            tree_txt = str(RenderTree(stmts))
            desc = generate_desc(i, sub_data, name=f"result{i}")
            tree_txt = f"// {desc}\n\n" + tree_txt
            with open(settings.out_dir / f"tree{i}.txt", "w") as f:
                f.write(tree_txt)
            index_artifacts[i]["tree"] = settings.out_dir / f"tree{i}.pkl"
        if settings.write.tree_fmt & ExportFormat.DOT:
            RenderTreeGraph(stmts, nodenamefunc=lambda x: x.summary).to_dotfile(settings.out_dir / f"tree{i}.dot")
        if settings.write.tree_fmt & ExportFormat.PNG:
            RenderTreeGraph(stmts, nodenamefunc=lambda x: x.summary).to_picture(settings.out_dir / f"tree{i}.png")
        if settings.write.tree_fmt & ExportFormat.PDF:
            RenderTreeGraph(stmts, nodenamefunc=lambda x: x.summary).to_picture(settings.out_dir / f"tree{i}.pdf")
