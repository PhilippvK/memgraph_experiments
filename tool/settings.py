"""Tool Settings."""
import logging
from pathlib import Path
from enum import IntFlag
from dataclasses import dataclass, asdict, fields, replace, field
from typing import List, Optional, Union

import yaml
import dacite
import networkx as nx
from anytree import AnyNode
from dacite import from_dict, Config

from .enums import ExportFormat, ExportFilter, InstrPredicate, CDFGStage, parse_enum_intflag
from .types import RegisterClass
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


def fix_enum_types(data):
    if "filters" in data:
        if "instr_predicates" in data["filters"]:
            data["filters"]["instr_predicates"] = parse_enum_intflag(data["filters"]["instr_predicates"], InstrPredicate)
        if "allowed_reg_classes" in data["filters"]:
            data["filters"]["allowed_reg_classes"] = parse_enum_intflag(data["filters"]["allowed_reg_classes"], RegisterClass)
    return data


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
        data = fix_enum_types(data)
        return cls.from_dict(data)

    @classmethod
    def from_yaml_file(cls, path: Path):
        """Parse settings from YAML file."""
        with open(path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        data = fix_enum_types(data)
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
                        assert t1 is t2, f"Type conflict ({t1} vs. {t2}))"
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
    xlen: Optional[int] = None


@dataclass
class MemgraphSettings(YAMLSettings):
    host: Optional[str] = None
    port: Optional[int] = None


@dataclass
class QuerySettings(YAMLSettings):
    session: Optional[str] = None
    func: Optional[str] = None
    bb: Optional[str] = None
    maxmisos: Optional[List] = None
    stage: Optional[int] = None
    limit_results: Optional[int] = None
    min_path_len: Optional[int] = None
    max_path_len: Optional[int] = None
    max_path_width: Optional[int] = None
    ignore_names: Optional[List[str]] = None
    ignore_op_types: Optional[List[str]] = None
    ignore_const_inputs: Optional[bool] = None


@dataclass
class FilterSettings(YAMLSettings):
    min_inputs: Optional[int] = None
    max_inputs: Optional[int] = None
    min_outputs: Optional[int] = None
    max_outputs: Optional[int] = None
    max_in_operands: Optional[int] = None
    max_out_operands: Optional[int] = None
    max_inout_operands: Optional[int] = None
    max_nodes: Optional[int] = None
    min_nodes: Optional[int] = None
    # TODO: MIN_FREQ, MIN_WEIGHT, MAX_INSTRS, MAX_UNIQUE_INSTRS
    allowed_enc_sizes: List[int] = None
    max_enc_footprint: Optional[float] = None
    max_enc_weight: Optional[float] = None
    min_enc_bits_left: Optional[int] = None
    min_iso_weight: Optional[float] = None
    max_loads: Optional[int] = None
    max_stores: Optional[int] = None
    max_mems: Optional[int] = None
    max_branches: Optional[int] = None
    # instr_predicates: Optional[Union[InstrPredicate, int]] = None
    instr_predicates: Optional[InstrPredicate] = None
    allowed_reg_classes: Optional[RegisterClass] = None

    # def get_instr_predicates(self, allow_none: bool = False):
    #     if self.instr_predicates is None:
    #         assert allow_none
    #         return None
    #     return parse_enum_intflag(self.instr_predicates, InstrPredicate)


@dataclass
class CommonWriteSettings(YAMLSettings):
    enable: Optional[bool] = None
    fmt: Optional[ExportFormat] = None
    flt: Optional[ExportFilter] = None


@dataclass
class WriteSubSettings(CommonWriteSettings):
    fmt: Optional[ExportFormat] = None
    flt: Optional[ExportFilter] = None


@dataclass
class WriteSettings(YAMLSettings):
    func: Optional[bool] = None
    func_fmt: ExportFormat = None
    # func_flt: ExportFilter = None
    sub: WriteSubSettings = field(default_factory=WriteSubSettings)
    io_sub: Optional[bool] = None
    io_sub_fmt: ExportFormat = None
    io_sub_flt: ExportFilter = None
    tree: Optional[bool] = None
    tree_fmt: ExportFormat = None
    tree_flt: ExportFilter = None
    gen: Optional[bool] = None
    gen_fmt: ExportFormat = None
    gen_flt: ExportFilter = None
    pie: Optional[bool] = None
    pie_fmt: ExportFormat = None
    pie_flt: ExportFilter = None
    sankey: Optional[bool] = None
    sankey_fmt: ExportFormat = None
    sankey_flt: ExportFilter = None
    df: Optional[bool] = None
    df_fmt: ExportFormat = None
    df_flt: ExportFilter = None
    index: Optional[bool] = None
    index_fmt: ExportFormat = None
    index_flt: ExportFilter = None
    queries: Optional[bool] = None
    query_metrics: Optional[bool] = None
    hdf5: Optional[bool] = None
    hdf5_flt: ExportFilter = None


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
    n_parallel: Optional[int] = None
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
    enable_variation_reuse_io: Optional[bool] = None
    enable_variation_auto_imm: Optional[bool] = None
    read_hdf5: Optional[bool] = None
    read_hdf5_flt: Optional[ExportFilter] = None

    @property
    def out_dir(self):
        return Path(self.out) if self.out is not None else None

    def get_defaults():
        settings = Settings(
            halt_on_error=defaults.HALT_ON_ERROR,
            until=defaults.UNTIL,
            filters=FilterSettings(
                min_inputs=defaults.MIN_INPUTS,
                max_inputs=defaults.MAX_INPUTS,
                min_outputs=defaults.MIN_OUTPUTS,
                max_outputs=defaults.MAX_OUTPUTS,
                max_in_operands=defaults.MAX_IN_OPERANDS,
                max_out_operands=defaults.MAX_OUT_OPERANDS,
                max_inout_operands=defaults.MAX_INOUT_OPERANDS,
                max_nodes=defaults.MAX_NODES,
                min_nodes=defaults.MIN_NODES,
                instr_predicates=defaults.INSTR_PREDICATES,
                allowed_reg_classes=defaults.ALLOWED_REG_CLASSES,
                allowed_enc_sizes=defaults.ALLOWED_ENC_SIZES,
                max_enc_footprint=defaults.MAX_ENC_FOOTPRINT,
                max_enc_weight=defaults.MAX_ENC_WEIGHT,
                min_enc_bits_left=defaults.MIN_ENC_BITS_LEFT,
                min_iso_weight=defaults.MIN_ISO_WEIGHT,
                max_loads=defaults.MAX_LOADS,
                max_stores=defaults.MAX_STORES,
                max_mems=defaults.MAX_MEMS,
                max_branches=defaults.MAX_BRANCHES,
            ),
            riscv=RISCVSettings(
                xlen=defaults.XLEN,
            ),
            memgraph=MemgraphSettings(
                host=defaults.HOST,
                port=defaults.PORT,
            ),
            query=QuerySettings(
                session=defaults.SESSION,
                func=defaults.FUNCTION,
                bb=defaults.BASIC_BLOCK,
                stage=CDFGStage(defaults.STAGE),
                limit_results=defaults.LIMIT_RESULTS,
                min_path_len=defaults.MIN_PATH_LENGTH,
                max_path_len=defaults.MAX_PATH_LENGTH,
                max_path_width=defaults.MAX_PATH_WIDTH,
                ignore_names=defaults.IGNORE_NAMES,
                ignore_op_types=defaults.IGNORE_OP_TYPES,
                ignore_const_inputs=defaults.IGNORE_CONST_INPUTS,
                maxmisos=defaults.MAXMISOS,
            ),
            write=WriteSettings(
                func=defaults.WRITE_FUNC,
                func_fmt=defaults.WRITE_FUNC_FMT,
                # func_flt=defaults.WRITE_FUNC_FLT,
                sub=WriteSubSettings(
                    enable=defaults.WRITE_SUB,
                    fmt=defaults.WRITE_SUB_FMT,
                    flt=defaults.WRITE_SUB_FLT,
                ),
                io_sub=defaults.WRITE_IO_SUB,
                io_sub_fmt=defaults.WRITE_IO_SUB_FMT,
                io_sub_flt=defaults.WRITE_IO_SUB_FLT,
                tree=defaults.WRITE_TREE,
                tree_fmt=defaults.WRITE_TREE_FMT,
                tree_flt=defaults.WRITE_TREE_FLT,
                gen=defaults.WRITE_GEN,
                gen_fmt=defaults.WRITE_GEN_FMT,
                gen_flt=defaults.WRITE_GEN_FLT,
                pie=defaults.WRITE_PIE,
                pie_fmt=defaults.WRITE_PIE_FMT,
                pie_flt=defaults.WRITE_PIE_FLT,
                sankey=defaults.WRITE_SANKEY,
                sankey_fmt=defaults.WRITE_SANKEY_FMT,
                sankey_flt=defaults.WRITE_SANKEY_FLT,
                df=defaults.WRITE_DF,
                df_fmt=defaults.WRITE_DF_FMT,
                df_flt=defaults.WRITE_DF_FLT,
                index=defaults.WRITE_INDEX,
                index_fmt=defaults.WRITE_INDEX_FMT,
                index_flt=defaults.WRITE_INDEX_FLT,
                queries=defaults.WRITE_QUERIES,
                query_metrics=defaults.WRITE_QUERY_METRICS,
                hdf5=defaults.WRITE_HDF5,
                hdf5_flt=defaults.WRITE_HDF5_FLT,
            ),
            # enable_variations=defaults.ENABLE_VARIATIONS,
            enable_variation_reuse_io=defaults.ENABLE_VARIATION_REUSE_IO,
            enable_variation_auto_imm=defaults.ENABLE_VARIATION_AUTO_IMM,
            read_hdf5=defaults.READ_HDF5,
            read_hdf5_flt=defaults.READ_HDF5_FLT,
        )
        return settings

    @staticmethod
    def initialize(args):
        logger.info("Initializing default settings")
        settings = Settings.get_defaults()
        # print("default_settings", settings)
        # print("default_settings.filters", settings.filters)
        # print("default_settings.filters.allowed_reg_classes", settings.filters.allowed_reg_classes)
        # input(">")
        if args.yaml is not None:
            logger.info("Initializing settings from yaml")
            settings_ = Settings.from_yaml_file(args.yaml)
            # print("yaml_settings", settings_)
            # print("yaml_settings.filters", settings_.filters)
            # print("yaml_settings.filters.allowed_reg_classes", settings_.filters.allowed_reg_classes)
            # input(">")
            settings = settings.merge(settings_)
        logger.info("Initializing settings from args")
        settings_ = Settings(
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
                max_in_operands=args.max_in_operands,
                max_out_operands=args.max_out_operands,
                max_nodes=args.max_nodes,
                min_nodes=args.min_nodes,
                instr_predicates=parse_enum_intflag(args.instr_predicates, InstrPredicate),
                allowed_reg_classes=parse_enum_intflag(args.allowed_reg_classes, RegisterClass),
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
                ignore_names=args.ignore_names.split(",") if args.ignore_names is not None else None,
                ignore_op_types=args.ignore_op_types.split(",") if args.ignore_op_types is not None else None,
                ignore_const_inputs=args.ignore_const_inputs,
                maxmisos=args.maxmisos.split(",") if args.maxmisos is not None else None,
            ),
            write=WriteSettings(
                func=args.write_func,
                func_fmt=args.write_func_fmt,
                # func_flt=args.write_func_flt,
                sub=WriteSubSettings(
                    enable=args.write_sub,
                    fmt=parse_enum_intflag(args.write_sub_fmt, ExportFormat) if args.write_sub_fmt is not None else None,
                    flt=parse_enum_intflag(args.write_sub_flt, ExportFilter) if args.write_sub_fmt is not None else None,
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
        # print("args_settings", settings_)
        # print("args_settings.filters", settings_.filters)
        # print("args_settings.filters.allowed_reg_classes", settings_.filters.allowed_reg_classes)
        # input(">")
        settings = settings.merge(settings_)
        return settings

    def validate(self):
        assert self.out_dir is not None
        assert self.out_dir.is_dir(), f"out ({self.out}) is not a directory"
        assert self.query.func is not None
        assert not self.query.ignore_const_inputs, "DEPRECTAED!"
        if self.until is not None:
            raise NotImplementedError("--until / stages")
