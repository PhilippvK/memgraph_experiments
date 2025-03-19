"""Tool Settings."""
import logging
from pathlib import Path
from enum import IntFlag
from dataclasses import dataclass, asdict, fields, replace, field
from typing import List, Optional

import yaml
import dacite
import networkx as nx
from anytree import AnyNode
from dacite import from_dict, Config

from .enums import ExportFormat, ExportFilter, InstrPredicate, CDFGStage, parse_enum_intflag
import tool.defaults as defaults

logger = logging.getLogger("settings")

ALLOWED_YAML_TYPES = (int, float, str, bool)


def check_supported_types(data):
    """Assert that now unsupported types are written into YAML file."""
    if isinstance(data, dict):
        for value in data.values():
            check_supported_types(value)
    elif isinstance(data, list):
        for x in data:
            check_supported_types(x)
    else:
        if data is not None:
            assert isinstance(data, ALLOWED_YAML_TYPES), f"Unsupported type: {type(data)}"


class YAMLSettings:  # TODO: make abstract
    """Generic YAMLSettings."""

    @classmethod
    def from_dict(cls, data: dict):
        """Convert dict into instance of YAMLSettings."""
        try:
            return from_dict(data_class=cls, data=data, config=Config(strict=True))
        except dacite.exceptions.UnexpectedDataError as err:
            logger.error("Unexpected key in Seal5Settings. Check for missmatch between Seal5 versions!")
            raise err

    @classmethod
    def from_yaml(cls, text: str):
        """Write settings to YAML file."""
        data = yaml.safe_load(text)
        return cls.from_dict(data)

    @classmethod
    def from_yaml_file(cls, path: Path):
        """Parse settings from YAML file."""
        with open(path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        return cls.from_dict(data)

    def to_yaml(self):
        """Convert settings to YAML string."""
        def dict_factory(fields):
            # print("dict_factory", fields)
            DROP_TYPES = (nx.MultiDiGraph, AnyNode)

            def fix_types(v):
                # INT_FLAG_TYPES = (CDFGStage, ExportFilter, ExportFormat, InstrPredicate)
                # if isinstance(v, INT_FLAG_TYPES):
                if isinstance(v, IntFlag):
                    return int(v)
                return v

            fields = [(k, fix_types(v)) for k, v in fields if not isinstance(v, DROP_TYPES)]
            # print("fields", fields)
            # input("111")
            return dict(fields)
        data = asdict(self, dict_factory=dict_factory)
        check_supported_types(data)
        text = yaml.dump(data)
        return text

    def to_yaml_file(self, path: Path):
        """Write settings to YAML file."""
        text = self.to_yaml()
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)

    def merge(self, other: "YAMLSettings", overwrite: bool = False, inplace: bool = False):
        """Merge two instances of YAMLSettings."""
        if not inplace:
            ret = replace(self)  # Make a copy of self
        for f1 in fields(other):
            k1 = f1.name
            v1 = getattr(other, k1)
            if v1 is None:
                continue
            t1 = type(v1)
            found = False
            for f2 in fields(self):
                k2 = f2.name
                v2 = getattr(self, k2)
                if k2 == k1:
                    found = True
                    if v2 is None:
                        if inplace:
                            setattr(self, k2, v1)
                        else:
                            setattr(ret, k2, v1)
                    else:
                        t2 = type(v2)
                        assert t1 is t2, "Type conflict"
                        if isinstance(v1, YAMLSettings):
                            v2.merge(v1, overwrite=overwrite, inplace=True)
                        elif isinstance(v1, dict):
                            if overwrite:
                                v2.clear()
                                v2.update(v1)
                            else:
                                for dict_key, dict_val in v1.items():
                                    if dict_key in v2:
                                        if isinstance(dict_val, YAMLSettings):
                                            assert isinstance(v2[dict_key], YAMLSettings)
                                            v2[dict_key].merge(dict_val, overwrite=overwrite, inplace=True)
                                        elif isinstance(dict_val, dict):
                                            v2[dict_key].update(dict_val)
                                        else:
                                            v2[dict_key] = dict_val
                                    else:
                                        v2[dict_key] = dict_val
                        elif isinstance(v1, list):
                            if overwrite:
                                v2.clear()
                            new = [x for x in v1 if x not in v2]
                            v2.extend(new)
                        else:
                            assert isinstance(
                                v2, (int, float, str, bool, Path)
                            ), f"Unsupported field type for merge {t1}"
                            if inplace:
                                setattr(self, k1, v1)
                            else:
                                setattr(ret, k1, v1)
                    break
            assert found
        if not inplace:
            return ret


@dataclass
class RISCVSettings(YAMLSettings):
    xlen: int = 64


@dataclass
class MemgraphSettings(YAMLSettings):
    host: str = "localhost"
    port: int = 7687


@dataclass
class QuerySettings(YAMLSettings):
    session: str = "default"
    func: Optional[str] = None
    bb: Optional[str] = None
    stage: int = defaults.STAGE_DEFAULT
    limit_results: Optional[int] = None
    min_path_len: int = 1
    max_path_len: int = 3
    max_path_width: int = 2
    ignore_names: List[str] = field(default_factory=defaults.IGNORE_NAMES_DEFAULT)
    ignore_op_types: List[str] = field(default_factory=defaults.IGNORE_OP_TYPES_DEFAULT)
    ignore_const_inputs: bool = False


@dataclass
class FilterSettings(YAMLSettings):
    min_inputs: int = 0
    max_inputs: int = 3
    min_outputs: int = 0
    max_outputs: int = 2
    max_nodes: int = 5
    min_nodes: int = 1
    # TODO: MIN_FREQ, MIN_WEIGHT, MAX_INSTRS, MAX_UNIQUE_INSTRS
    allowed_enc_sizes: List[int] = field(default_factory=defaults.ALLOWED_ENC_SIZES_DEFAULT)
    max_enc_footprint: float = defaults.MAX_ENC_FOOTPRINT_DEFAULT
    max_enc_weight: float = defaults.MAX_ENC_WEIGHT_DEFAULT
    min_enc_bits_left: int = defaults.MIN_ENC_BITS_LEFT_DEFAULT
    min_iso_weight: float = defaults.MIN_ISO_WEIGHT_DEFAULT
    max_loads: int = defaults.MAX_LOADS_DEFAULT
    max_stores: int = defaults.MAX_STORES_DEFAULT
    max_mems: int = defaults.MAX_MEMS_DEFAULT
    max_branches: int = defaults.MAX_BRANCHES_DEFAULT
    instr_predicates: InstrPredicate = defaults.INSTR_PREDICATES_DEFAULT


@dataclass
class CommonWriteSettings(YAMLSettings):
    enable: bool = False
    fmt: ExportFormat = ExportFormat.NONE
    flt: ExportFilter = ExportFilter.NONE


@dataclass
class WriteSubSettings(CommonWriteSettings):
    fmt: ExportFormat = defaults.SUB_FMT_DEFAULT
    flt: ExportFilter = defaults.SUB_FLT_DEFAULT


@dataclass
class WriteSettings(YAMLSettings):
    func: bool = False
    func_fmt: ExportFormat = defaults.SUB_FMT_DEFAULT
    # func_flt: ExportFilter = defaults.SUB_FLT_DEFAULT
    sub: WriteSubSettings = field(default_factory=WriteSubSettings)
    io_sub: bool = False
    io_sub_fmt: ExportFormat = defaults.IO_SUB_FMT_DEFAULT
    io_sub_flt: ExportFilter = defaults.IO_SUB_FLT_DEFAULT
    tree: bool = False
    tree_fmt: ExportFormat = defaults.TREE_FMT_DEFAULT
    tree_flt: ExportFilter = defaults.TREE_FLT_DEFAULT
    gen: bool = False
    gen_fmt: ExportFormat = defaults.GEN_FMT_DEFAULT
    gen_flt: ExportFilter = defaults.GEN_FLT_DEFAULT
    pie: bool = False
    pie_fmt: ExportFormat = defaults.PIE_FMT_DEFAULT
    pie_flt: ExportFilter = defaults.PIE_FLT_DEFAULT
    sankey: bool = False
    sankey_fmt: ExportFormat = defaults.SANKEY_FMT_DEFAULT
    sankey_flt: ExportFilter = defaults.SANKEY_FLT_DEFAULT
    df: bool = False
    df_fmt: ExportFormat = defaults.DF_FMT_DEFAULT
    df_flt: ExportFilter = defaults.DF_FLT_DEFAULT
    index: bool = False
    index_fmt: ExportFormat = defaults.INDEX_FMT_DEFAULT
    index_flt: ExportFilter = defaults.INDEX_FLT_DEFAULT
    queries: bool = False
    query_metrics: bool = False
    hdf5: bool = False
    hdf5_flt: ExportFilter = defaults.WRITE_HDF5_FLT_DEFAULT


@dataclass
class Settings(YAMLSettings):
    """Tool settings."""
    # TODO: define other defaults
    # TODO: allow passing settings via --yaml
    # TODO: do not overwrite with optional args
    # TODO: enable WRITE by default
    # TODO: expose --no-write-... args

    # General settings
    out: Optional[str] = None
    progress: bool = False
    times: bool = False
    append_times: bool = False
    halt_on_error: bool = False
    until: Optional[str] = None  # TODO: replace

    # RISC-V settings
    riscv: RISCVSettings = field(default_factory=RISCVSettings)

    # Filter settings
    filters: FilterSettings = field(default_factory=FilterSettings)

    # Memgraph settings
    memgraph: MemgraphSettings = field(default_factory=MemgraphSettings)

    # Query settings
    query: QuerySettings = field(default_factory=QuerySettings)

    # Write settings
    write: WriteSettings = field(default_factory=WriteSettings)

    # Rest
    # enable_variations = False
    enable_variation_reuse_io: bool = False
    read_hdf5: bool = False
    read_hdf5_flt: ExportFilter = defaults.READ_HDF5_FLT_DEFAULT

    @property
    def out_dir(self):
        return Path(self.out) if self.out is not None else None

    @staticmethod
    def initialize(args):
        if args.yaml is not None:
            logger.info("Initializing settings from yaml")
            raise NotImplementedError
        else:
            logger.info("Initializing settings from args")
            settings = Settings(
                out=str(Path(args.output_dir).resolve()),
                progress=args.progress,
                times=args.times,
                append_times=args.append_times,
                halt_on_error=args.halt_on_error,
                until=args.until,
                filters=FilterSettings(
                    min_inputs=args.min_inputs,
                    max_inputs=args.max_inputs,
                    min_outputs=args.min_outputs,
                    max_outputs=args.max_outputs,
                    max_nodes=args.max_nodes,
                    min_nodes=args.min_nodes,
                    instr_predicates=parse_enum_intflag(args.instr_predicates, InstrPredicate),
                    allowed_enc_sizes=args.allowed_enc_sizes,
                    max_enc_footprint=args.max_enc_footprint,
                    max_enc_weight=args.max_enc_weight,
                    min_enc_bits_left=args.min_enc_bits_left,
                    min_iso_weight=args.min_iso_weight,
                    max_loads=args.max_loads,
                    max_stores=args.max_stores,
                    max_mems=args.max_mems,
                    max_branches=args.max_branches,
                ),
                riscv=RISCVSettings(
                    xlen=args.xlen,
                ),
                memgraph=MemgraphSettings(
                    host=args.host,
                    port=args.port,
                ),
                query=QuerySettings(
                    session=args.session,
                    func=args.function,
                    bb=args.basic_block,
                    stage=CDFGStage(args.stage),
                    limit_results=args.limit_results,
                    min_path_len=args.min_path_length,
                    max_path_len=args.max_path_length,
                    max_path_width=args.max_path_width,
                    ignore_names=args.ignore_names.split(","),
                    ignore_op_types=args.ignore_op_types.split(","),
                    ignore_const_inputs=args.ignore_const_inputs,
                ),
                write=WriteSettings(
                    func=args.write_func,
                    func_fmt=args.write_func_fmt,
                    # func_flt=args.write_func_flt,
                    sub=WriteSubSettings(
                        enable=args.write_sub,
                        fmt=parse_enum_intflag(args.write_sub_fmt, ExportFormat),
                        flt=parse_enum_intflag(args.write_sub_flt, ExportFilter),
                    ),
                    io_sub=args.write_io_sub,
                    io_sub_fmt=args.write_io_sub_fmt,
                    io_sub_flt=args.write_io_sub_flt,
                    tree=args.write_tree,
                    tree_fmt=args.write_tree_fmt,
                    tree_flt=args.write_tree_flt,
                    gen=args.write_gen,
                    gen_fmt=args.write_gen_fmt,
                    gen_flt=args.write_gen_flt,
                    pie=args.write_pie,
                    pie_fmt=args.write_pie_fmt,
                    pie_flt=args.write_pie_flt,
                    sankey=args.write_sankey,
                    sankey_fmt=args.write_sankey_fmt,
                    sankey_flt=args.write_sankey_flt,
                    df=args.write_df,
                    df_fmt=args.write_df_fmt,
                    df_flt=args.write_df_flt,
                    index=args.write_index,
                    index_fmt=args.write_index_fmt,
                    index_flt=args.write_index_flt,
                    queries=args.write_queries,
                    query_metrics=args.write_query_metrics,
                    hdf5=args.write_hdf5,
                    hdf5_flt=args.write_hdf5_flt,
                ),
                # enable_variations=args.enable_variations,
                enable_variation_reuse_io=args.enable_variation_reuse_io,
                read_hdf5=args.read_hdf5,
                read_hdf5_flt=args.read_hdf5_flt,
            )
        return settings

    def validate(self):
        assert self.out_dir is not None
        assert self.out_dir.is_dir(), f"out ({self.out}) is not a directory"
        assert self.query.func is not None
        assert not self.query.ignore_const_inputs, "DEPRECTAED!"
        if self.until is not None:
            raise NotImplementedError("--until / stages")
