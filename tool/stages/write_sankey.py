import logging

# import h5py
# import numpy as np


from ..enums import ExportFilter, ExportFormat, Variation
from ..pie import pie_name_helper, pie_name_helper2

logger = logging.getLogger("write_hdf5")


def write_hdf5(settings, subs_df, index_artifacts):
    logger.info("Exporting Sankey chart...")
    if settings.write.sankey_fmt & ExportFormat.MARKDOWN:  # MERMAID?
        # TODO: handle variations
        filtered_subs_df = subs_df[(subs_df["Status"] & settings.write.sankey_flt) > 0].copy()
        counts_df = filtered_subs_df.value_counts("Status").rename_axis("Status").reset_index(name="Count")
        # counts_df2 = filtered_subs_df["Parent"].apply(lambda x: "Original" if pd.isna(x) else "Variation").value_counts().rename_axis("Parent").reset_index(name="Count")
        counts_df2 = filtered_subs_df["Variations"].value_counts().rename_axis("Parent").reset_index(name="Count")

        # print("counts_df", counts_df)
        # print("counts_df2", counts_df2)
        sankey_data = []
        total = counts_df["Count"].sum()
        counts_dict = counts_df.set_index("Status")["Count"].to_dict()
        counts_dict2 = counts_df2.set_index("Parent")["Count"].to_dict()
        # print("counts_dict", counts_dict)
        levels = {
            "temp1": ([ExportFilter.ISO], []),
            "temp2": (
                [
                    ExportFilter.FILTERED_IO,
                    ExportFilter.FILTERED_COMPLEX,
                    ExportFilter.FILTERED_SIMPLE,
                    ExportFilter.FILTERED_PRED,
                    ExportFilter.FILTERED_MEM,
                    ExportFilter.FILTERED_BRANCH,
                    ExportFilter.INVALID,
                ],
                [],
            ),
            "temp3": ([ExportFilter.FILTERED_OPERANDS], [Variation.REUSE_IO]),  # TODO: make variation enum
            "temp4": ([ExportFilter.FILTERED_ENC], []),
            "temp5": ([ExportFilter.FILTERED_WEIGHTS], []),
            "temp6": ([ExportFilter.ERROR], []),
            "temp7": ([ExportFilter.SELECTED], []),
        }
        current = "query"
        for level_name, temp in levels.items():
            filters, variations = temp
            # print("level_name", level_name)
            # print("filters", filters)
            # print("variation", variations)
            # print("current", current)
            # print("total", total)
            total_new = total
            for var in variations:
                # print("var", var)
                var_count = counts_dict2.get(var, 0)
                # print("var_count", var_count)
                if var_count > 0:
                    new = (pie_name_helper2(var), current, var_count)
                    # print("new", new)
                    sankey_data.append(new)
                total_new += var_count
            for flt in filters:
                # print("flt", flt)
                flt_count = counts_dict.get(flt, 0)
                # print("flt_count", flt_count)
                if flt_count > 0:
                    new = (current, pie_name_helper(flt), flt_count)
                    # print("new", new)
                    sankey_data.append(new)
                total_new -= flt_count
            assert total_new >= 0
            if total_new != 0:
                new = (current, level_name, total_new)
                # print("new", new)
                sankey_data.append(new)
            current = level_name
            total = total_new
        # print("sankey_data", sankey_data)
        content = """```mermaid
---
config:
  sankey:
    showValues: true
---
sankey-beta

%% source,target,value
"""
        for source, target, value in sankey_data:
            content += f"{source},{target},{value}\n"
        content += "```"
        with open(settings.out_dir / "sankey.md", "w") as f:
            f.write(content)
        index_artifacts[None]["sankey"] = settings.out_dir / "sankey.md"
