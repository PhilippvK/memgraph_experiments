from pathlib import Path
from typing import Union, Dict, Any

import yaml
import pandas as pd


def write_index_file(dest: Union[str, Path], subs_df: pd.DataFrame, index_data: Dict[int, Dict[str, Any]]):
    yaml_data = {"candidates": []}
    for sub_id in subs_df.index:
        assert sub_id in index_data

        def helper(x):
            if isinstance(x, Path):
                x = str(x)
            return x

        artifacts_data = {key: helper(value) for key, value in index_data[sub_id].items()}
        new = {"id": sub_id, "artifacts": artifacts_data}
        yaml_data["candidates"].append(new)
    with open(dest, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
