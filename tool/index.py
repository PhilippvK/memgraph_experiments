from pathlib import Path
from typing import Union, Dict, Any, Optional, List
from dataclasses import dataclass, field

import yaml
import pandas as pd
import numpy as np
import networkx as nx
from anytree import AnyNode

from .settings import Settings, YAMLSettings


def yaml_types_helper(x, allow_missing: bool = False):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (int, float, str, list)):
        return x
    if isinstance(x, (set, tuple)):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if not allow_missing:
        assert NotImplementedError("Unsupported Type: {type(x)}")
    return x


@dataclass
class Candidate(YAMLSettings):
    sub: Optional[nx.MultiDiGraph] = None  # TODO: do not write to yaml
    io_sub: Optional[nx.MultiDiGraph] = None  # TODO: do not write to yaml
    tree: Optional[AnyNode] = None  # TODO: do not write to yaml
    artifacts: Dict[str , str] = field(default_factory=dict)
    properties: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    index: Optional[int] = None
    # TODO: add metrics


@dataclass
class Index(YAMLSettings):
    func_graph: Optional[nx.MultiDiGraph] = None  # TODO: do not write to yaml
    candidates_graph: Optional[nx.MultiDiGraph] = None  # TODO: do not write to yaml
    artifacts: Dict[str , str] = field(default_factory=dict)
    settings: Optional[Settings] = None
    properties: Optional[Dict[str, Any]] = None  # Redundant with stettings?
    metrics: Optional[Dict[str, Any]] = None
    candidates: List[Candidate] = field(default_factory=list)
    # TODO: add metrics


@dataclass
class MultiIndex(YAMLSettings):
    artifacts: List[Dict[str , str]] = field(default_factory=list)
    settings: List[Optional[Settings]] = None
    properties: List[Optional[Dict[str, Any]]] = None  # Redundant with stettings?
    candidates: List[Candidate] = field(default_factory=list)
    # TODO: add metrics


def write_index_file(
    settings: Settings,
    dest: Union[str, Path],
    GF,
    G,
    subs,
    io_subs,
    trees,
    subs_df: pd.DataFrame,
    global_df: pd.DataFrame,
    index_artifacts: Dict[int, Dict[str, Any]],
    include_properties: bool = True,
):
    yaml_data = {"global": {}, "candidates": []}
    global_artifacts_data = {key: yaml_types_helper(value) for key, value in index_artifacts[None].items()}
    yaml_data["global"]["artifacts"] = global_artifacts_data
    yaml_data["global"]["properties"] = []
    yaml_data["global"]["metrics"] = []
    if include_properties:
        # assert len(global_df) == 1
        # row = global_df.iloc[0]
        for index, row in global_df.iterrows():
            row_data = {key: yaml_types_helper(value) for key, value in row.to_dict().items()}
            # print("row_data", row_data)
            yaml_data["global"]["properties"].append(row_data)
    for sub_id, row in subs_df.iterrows():
        assert sub_id in index_artifacts

        artifacts_data = {key: yaml_types_helper(value) for key, value in index_artifacts[sub_id].items()}
        new = {"id": sub_id, "artifacts": artifacts_data, "metrics": {}}

        if include_properties:

            row_data = {key: yaml_types_helper(value) for key, value in row.to_dict().items()}
            # print("row_data", row_data)
            new["properties"] = row_data

        yaml_data["candidates"].append(new)
    with open(dest, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    # TODO: use Index in main.py
    index = Index(
        func_graph=GF,
        candidates_graph=G,
        artifacts=global_artifacts_data,
        settings=settings,
        properties=yaml_data["global"]["properties"][0],
        # TODO: metrics
        candidates=[
            Candidate(
                sub=subs[c["id"]],
                io_sub=io_subs[c["id"]],
                tree=trees[c["id"]],
                artifacts=c["artifacts"],
                properties=c["properties"],
            )
            for c in yaml_data["candidates"]
        ],
    )
    TEMP_PATH = "/tmp/abc.yml"
    index.to_yaml(TEMP_PATH)
    print("index", index)
    input(">>>")
