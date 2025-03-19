from .enums import ExportFormat, ExportFilter, InstrPredicate, CDFGStage

# TODO: rename result to sub & gen
# TODO: actually implement filters
FUNC_FMT_DEFAULT = ExportFormat.DOT | ExportFormat.PKL
# FUNC_FLT_DEFAULT = ExportFilter.SELECTED
# SUB_FMT_DEFAULT = ExportFormat.DOT  # | ExportFormat.PDF | ExportFormat.PNG
# SUB_FMT_DEFAULT = ExportFormat.DOT | ExportFormat.PKL  # | ExportFormat.PNG | ExportFormat.PDF
SUB_FMT_DEFAULT = ExportFormat.DOT | ExportFormat.PKL | ExportFormat.PDF
SUB_FLT_DEFAULT = ExportFilter.SELECTED
# SUB_FLT_DEFAULT = ExportFilter.ALL
# IO_SUB_FMT_DEFAULT = ExportFormat.DOT | ExportFormat.PKL  # | ExportFormat.PNG | ExportFormat.PDF
IO_SUB_FMT_DEFAULT = ExportFormat.DOT | ExportFormat.PKL | ExportFormat.PDF
IO_SUB_FLT_DEFAULT = ExportFilter.SELECTED
# TODO: move Tree tro pre-gen
TREE_FMT_DEFAULT = ExportFormat.TXT | ExportFormat.PKL | ExportFormat.DOT | ExportFormat.PDF
TREE_FLT_DEFAULT = ExportFilter.SELECTED
# GEN_FMT_DEFAULT = ExportFormat.CDSL | ExportFormat.MIR | ExportFormat.FLAT
GEN_FMT_DEFAULT = ExportFormat.FLAT | ExportFormat.CDSL
GEN_FLT_DEFAULT = ExportFilter.SELECTED
PIE_FMT_DEFAULT = ExportFormat.PDF | ExportFormat.CSV | ExportFormat.PNG
PIE_FLT_DEFAULT = ExportFilter.ALL
SANKEY_FMT_DEFAULT = ExportFormat.MARKDOWN
SANKEY_FLT_DEFAULT = ExportFilter.ALL & ~ExportFilter.ISO
DF_FMT_DEFAULT = ExportFormat.CSV | ExportFormat.PKL
DF_FLT_DEFAULT = ExportFilter.ALL
INDEX_FMT_DEFAULT = ExportFormat.YAML
INDEX_FLT_DEFAULT = ExportFilter.SELECTED
WRITE_HDF5_FLT_DEFAULT = ExportFilter.SELECTED | ExportFilter.FILTERED_WEIGHTS | ExportFilter.ERROR | ExportFilter.INVALID
READ_HDF5_FLT_DEFAULT = ExportFilter.SELECTED
INSTR_PREDICATES_DEFAULT = InstrPredicate.ALL
# IGNORE_NAMES_DEFAULT = ["G_PHI", "PHI", "COPY", "PseudoCALLIndirect", "PseudoLGA", "Select_GPR_Using_CC_GPR"]
IGNORE_NAMES_DEFAULT = ["G_PHI", "PHI", "COPY", "PseudoCALLIndirect", "PseudoLGA", "Select_GPR_Using_CC_GPR", "LUI", "PseudoMovAddr"]
IGNORE_OP_TYPES_DEFAULT = ["input", "constant", "label"]  # TODO: allow_label? (handle frame_index for stack?)
ALLOWED_ENC_SIZES_DEFAULT = [32]
STAGE_DEFAULT = CDFGStage.STAGE_3
MAX_ENC_FOOTPRINT_DEFAULT = 1.0
MAX_ENC_WEIGHT_DEFAULT = 1.0
MIN_ENC_BITS_LEFT_DEFAULT = 5
MIN_ISO_WEIGHT_DEFAULT = 0.05
MAX_LOADS_DEFAULT = 1
MAX_STORES_DEFAULT = 1
MAX_MEMS_DEFAULT = 1
MAX_BRANCHES_DEFAULT = 1
ALLOWED_IMM_WIDTHS = [5, 12]
MAX_OPERANDS_DEFAULT = 5
MAX_REG_OPERANDS_DEFAULT = 5
MAX_IMM_OPERANDS_DEFAULT = 2
