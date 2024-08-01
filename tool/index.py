from pathlib import Path
from typing import Union, Dict, Any

import yaml
import pandas as pd


def yaml_types_helper(x, allow_missing: bool = False):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (int, float, str, list)):
        return x
    if isinstance(x, (set, tuple)):
        return list(x)
    if not allow_missing:
        assert NotImplementedError("Unsupported Type: {type(x)}")
    return x


def write_index_file(
    dest: Union[str, Path],
    subs_df: pd.DataFrame,
    global_df: pd.DataFrame,
    index_artifacts: Dict[int, Dict[str, Any]],
    include_properties: bool = True,
):
    yaml_data = {"global": {}, "candidates": []}
    global_artifacts_data = {key: yaml_types_helper(value) for key, value in index_artifacts[None].items()}
    yaml_data["global"]["artifacts"] = global_artifacts_data
    yaml_data["global"]["properties"] = []
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
        new = {"id": sub_id, "artifacts": artifacts_data}

        if include_properties:

            row_data = {key: yaml_types_helper(value) for key, value in row.to_dict().items()}
            # print("row_data", row_data)
            new["properties"] = row_data

        yaml_data["candidates"].append(new)
    with open(dest, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
