import logging

from tqdm import tqdm

from ..types import RegisterClass

logger = logging.getLogger("detect_registers")


MAPPING = {
    "gpr": RegisterClass.GPR,
    "fpr": RegisterClass.FPR,
    "fpr32": RegisterClass.FPR,  # TODO: get flen?
    "fpr64": RegisterClass.FPR,  # TODO: get flen?
    "csr": RegisterClass.CSR,
    "unknown": RegisterClass.UNKNOWN,
}


def collect_register_classes(sub):
    ret = []
    for node in sub.nodes:
        node_data = sub.nodes[node]
        properties = node_data["properties"]
        out_reg_class_name = properties.get("out_reg_class")
        print(out_reg_class_name)
        out_reg_class = MAPPING.get(out_reg_class_name, RegisterClass.UNKNOWN)
        ret.append(out_reg_class)
    return ret


def detect_registers(settings, subs, subs_df, io_isos):
    logger.info("Detecting Registers...")
    subs_df["Registers"] = RegisterClass.NONE
    subs_iter = [(i, sub) for i, sub in enumerate(subs) if i not in io_isos]
    # for i, sub in enumerate(tqdm(subs, disable=not settings.progress)):
    for i, sub in tqdm(subs_iter, disable=not settings.progress):
        reg_classes = collect_register_classes(sub)
        # print("reg_classes", reg_classes)
        regs_flag = RegisterClass.NONE
        for reg_class in reg_classes:
            regs_flag |= reg_class
        # print("regs_flag", regs_flag)
        subs_df.loc[i, "Registers"] = regs_flag
        # input("!")
