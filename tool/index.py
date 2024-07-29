from pathlib import Path
from typing import Union, Dict, Any

import yaml
import pandas as pd


def write_index_file(
    dest: Union[str, Path],
    subs_df: pd.DataFrame,
    index_data: Dict[int, Dict[str, Any]],
    include_properties: bool = True,
):
    yaml_data = {"global": {}, "candidates": []}
    yaml_data["global"]["artifacts"] = index_data[None]
    for sub_id, row in subs_df.iterrows():
        assert sub_id in index_data

        def helper(x):
            if isinstance(x, Path):
                x = str(x)
            return x

        artifacts_data = {key: helper(value) for key, value in index_data[sub_id].items()}
        new = {"id": sub_id, "artifacts": artifacts_data}

        if include_properties:

            def fix_types(x):
                if isinstance(x, (int, float, str, list)):
                    return x
                if isinstance(x, (set, tuple)):
                    return list(x)
                assert NotImplementedError("Unsupported Type: {type(x)}")

            row_data = {key: fix_types(value) for key, value in row.to_dict().items()}
            # print("row_data", row_data)
            new["properties"] = row_data
        yaml_data["candidates"].append(new)
    with open(dest, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
